/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent;

import static org.opensearch.common.xcontent.json.JsonXContent.jsonXContent;
import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.CommonValue.MCP_CONNECTORS_FIELD;
import static org.opensearch.ml.common.CommonValue.ML_AGENT_INDEX;
import static org.opensearch.ml.common.CommonValue.ML_TASK_INDEX;
import static org.opensearch.ml.common.MLTask.RESPONSE_FIELD;
import static org.opensearch.ml.common.MLTask.STATE_FIELD;
import static org.opensearch.ml.common.MLTask.TASK_ID_FIELD;
import static org.opensearch.ml.common.output.model.ModelTensorOutput.INFERENCE_RESULT_FIELD;
import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_MCP_CONNECTOR_DISABLED_MESSAGE;
import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_MCP_CONNECTOR_ENABLED;
import static org.opensearch.ml.common.utils.MLTaskUtils.updateMLTaskDirectly;

import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import org.opensearch.ExceptionsHelper;
import org.opensearch.OpenSearchException;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.ResourceNotFoundException;
import org.opensearch.action.get.GetResponse;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.Strings;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLAgentType;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.MLTaskType;
import org.opensearch.ml.common.agent.MLAgent;
import org.opensearch.ml.common.agent.MLMemorySpec;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.Input;
import org.opensearch.ml.common.input.execute.agent.AgentMLInput;
import org.opensearch.ml.common.output.MLTaskOutput;
import org.opensearch.ml.common.output.Output;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.settings.SettingsChangeListener;
import org.opensearch.ml.common.spi.memory.Memory;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.engine.Executable;
import org.opensearch.ml.engine.annotation.Function;
import org.opensearch.ml.engine.encryptor.Encryptor;
import org.opensearch.ml.engine.memory.ConversationIndexMemory;
import org.opensearch.ml.engine.memory.ConversationIndexMessage;
import org.opensearch.ml.memory.action.conversation.CreateInteractionResponse;
import org.opensearch.ml.memory.action.conversation.GetInteractionAction;
import org.opensearch.ml.memory.action.conversation.GetInteractionRequest;
import org.opensearch.remote.metadata.client.GetDataObjectRequest;
import org.opensearch.remote.metadata.client.PutDataObjectRequest;
import org.opensearch.remote.metadata.client.SdkClient;
import org.opensearch.remote.metadata.common.SdkClientUtils;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.transport.client.Client;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.gson.Gson;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;

/**
 * MLAgentExecutor is responsible for executing ML agents with various types and configurations.
 * It handles agent retrieval, memory management, task execution, and output processing.
 * 
 * Key responsibilities:
 * - Validate input and security permissions
 * - Retrieve and parse ML agents from the index
 * - Manage conversation memory and interactions
 * - Execute agents synchronously or asynchronously
 * - Process and format agent outputs
 */
@Log4j2
@Data
@NoArgsConstructor
@Function(FunctionName.AGENT)
public class MLAgentExecutor implements Executable, SettingsChangeListener {

    // Parameter constants for better maintainability
    public static final String MEMORY_ID = "memory_id";
    public static final String QUESTION = "question";
    public static final String PARENT_INTERACTION_ID = "parent_interaction_id";
    public static final String REGENERATE_INTERACTION_ID = "regenerate_interaction_id";
    public static final String MESSAGE_HISTORY_LIMIT = "message_history_limit";
    public static final String ERROR_MESSAGE = "error_message";

    // Core dependencies injected via constructor
    private Client client;
    private SdkClient sdkClient;
    private Settings settings;
    private ClusterService clusterService;
    private NamedXContentRegistry xContentRegistry;
    private Map<String, Tool.Factory> toolFactories;
    private Map<String, Memory.Factory> memoryFactoryMap;
    private volatile Boolean isMultiTenancyEnabled;
    private Encryptor encryptor;
    private static volatile boolean mcpConnectorIsEnabled;

    public MLAgentExecutor(
        Client client,
        SdkClient sdkClient,
        Settings settings,
        ClusterService clusterService,
        NamedXContentRegistry xContentRegistry,
        Map<String, Tool.Factory> toolFactories,
        Map<String, Memory.Factory> memoryFactoryMap,
        Boolean isMultiTenancyEnabled,
        Encryptor encryptor
    ) {
        this.client = client;
        this.sdkClient = sdkClient;
        this.settings = settings;
        this.clusterService = clusterService;
        this.xContentRegistry = xContentRegistry;
        this.toolFactories = toolFactories;
        this.memoryFactoryMap = memoryFactoryMap;
        this.isMultiTenancyEnabled = isMultiTenancyEnabled;
        this.encryptor = encryptor;
        this.mcpConnectorIsEnabled = ML_COMMONS_MCP_CONNECTOR_ENABLED.get(clusterService.getSettings());
        clusterService.getClusterSettings().addSettingsUpdateConsumer(ML_COMMONS_MCP_CONNECTOR_ENABLED, it -> mcpConnectorIsEnabled = it);
    }

    @Override
    public void onMultiTenancyEnabledChanged(boolean isEnabled) {
        this.isMultiTenancyEnabled = isEnabled;
    }

    @Override
    public void execute(Input input, ActionListener<Output> listener) {
        if (!(input instanceof AgentMLInput)) {
            throw new IllegalArgumentException("wrong input");
        }
        AgentMLInput agentMLInput = (AgentMLInput) input;
        String agentId = agentMLInput.getAgentId();
        String tenantId = agentMLInput.getTenantId();
        Boolean isAsync = agentMLInput.getIsAsync();

        RemoteInferenceInputDataSet inputDataSet = (RemoteInferenceInputDataSet) agentMLInput.getInputDataset();
        if (inputDataSet == null || inputDataSet.getParameters() == null) {
            throw new IllegalArgumentException("Agent input data can not be empty.");
        }

        if (isMultiTenancyEnabled && tenantId == null) {
            throw new OpenSearchStatusException("You don't have permission to access this resource", RestStatus.FORBIDDEN);
        }

        List<ModelTensors> outputs = new ArrayList<>();
        List<ModelTensor> modelTensors = new ArrayList<>();
        outputs.add(ModelTensors.builder().mlModelTensors(modelTensors).build());

        FetchSourceContext fetchSourceContext = new FetchSourceContext(true, Strings.EMPTY_ARRAY, Strings.EMPTY_ARRAY);
        GetDataObjectRequest getDataObjectRequest = GetDataObjectRequest
            .builder()
            .index(ML_AGENT_INDEX)
            .id(agentId)
            .tenantId(tenantId)
            .fetchSourceContext(fetchSourceContext)
            .build();

        if (clusterService.state().metadata().hasIndex(ML_AGENT_INDEX)) {
            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                sdkClient
                    .getDataObjectAsync(getDataObjectRequest, client.threadPool().executor("opensearch_ml_general"))
                    .whenComplete((response, throwable) -> {
                        context.restore();
                        log.debug("Completed Get Agent Request, Agent id:{}", agentId);
                        if (throwable != null) {
                            Exception cause = SdkClientUtils.unwrapAndConvertToException(throwable);
                            if (ExceptionsHelper.unwrap(cause, IndexNotFoundException.class) != null) {
                                log.error("Failed to get Agent index", cause);
                                listener.onFailure(new OpenSearchStatusException("Failed to get agent index", RestStatus.NOT_FOUND));
                            } else {
                                log.error("Failed to get ML Agent {}", agentId, cause);
                                listener.onFailure(cause);
                            }
                        } else {
                            try {
                                GetResponse getAgentResponse = response.parser() == null
                                    ? null
                                    : GetResponse.fromXContent(response.parser());
                                if (getAgentResponse != null && getAgentResponse.isExists()) {
                                    try (
                                        XContentParser parser = jsonXContent
                                            .createParser(
                                                xContentRegistry,
                                                LoggingDeprecationHandler.INSTANCE,
                                                getAgentResponse.getSourceAsString()
                                            )
                                    ) {
                                        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                                        MLAgent mlAgent = MLAgent.parse(parser);
                                        if (isMultiTenancyEnabled && !Objects.equals(tenantId, mlAgent.getTenantId())) {
                                            listener
                                                .onFailure(
                                                    new OpenSearchStatusException(
                                                        "You don't have permission to access this resource",
                                                        RestStatus.FORBIDDEN
                                                    )
                                                );
                                        }
                                        MLMemorySpec memorySpec = mlAgent.getMemory();
                                        String memoryId = inputDataSet.getParameters().get(MEMORY_ID);
                                        String parentInteractionId = inputDataSet.getParameters().get(PARENT_INTERACTION_ID);
                                        String regenerateInteractionId = inputDataSet.getParameters().get(REGENERATE_INTERACTION_ID);
                                        String appType = mlAgent.getAppType();
                                        String question = inputDataSet.getParameters().get(QUESTION);

                                        MLTask mlTask = MLTask
                                            .builder()
                                            .taskType(MLTaskType.AGENT_EXECUTION)
                                            .functionName(FunctionName.AGENT)
                                            .state(MLTaskState.CREATED)
                                            .workerNodes(ImmutableList.of(clusterService.localNode().getId()))
                                            .createTime(Instant.now())
                                            .lastUpdateTime(Instant.now())
                                            .async(false)
                                            .tenantId(tenantId)
                                            .build();

                                        if (memoryId == null && regenerateInteractionId != null) {
                                            throw new IllegalArgumentException("A memory ID must be provided to regenerate.");
                                        }
                                        if (memorySpec != null
                                            && memorySpec.getType() != null
                                            && memoryFactoryMap.containsKey(memorySpec.getType())
                                            && (memoryId == null || parentInteractionId == null)) {
                                            ConversationIndexMemory.Factory conversationIndexMemoryFactory =
                                                (ConversationIndexMemory.Factory) memoryFactoryMap.get(memorySpec.getType());
                                            conversationIndexMemoryFactory
                                                .create(question, memoryId, appType, ActionListener.wrap(memory -> {
                                                    inputDataSet.getParameters().put(MEMORY_ID, memory.getConversationId());
                                                    // get question for regenerate
                                                    if (regenerateInteractionId != null) {
                                                        log.info("Regenerate for existing interaction {}", regenerateInteractionId);
                                                        client
                                                            .execute(
                                                                GetInteractionAction.INSTANCE,
                                                                new GetInteractionRequest(regenerateInteractionId),
                                                                ActionListener.wrap(interactionRes -> {
                                                                    inputDataSet
                                                                        .getParameters()
                                                                        .putIfAbsent(QUESTION, interactionRes.getInteraction().getInput());
                                                                    saveRootInteractionAndExecute(
                                                                        listener,
                                                                        memory,
                                                                        inputDataSet,
                                                                        mlTask,
                                                                        isAsync,
                                                                        outputs,
                                                                        modelTensors,
                                                                        mlAgent
                                                                    );
                                                                }, e -> {
                                                                    log.error("Failed to get existing interaction for regeneration", e);
                                                                    listener.onFailure(e);
                                                                })
                                                            );
                                                    } else {
                                                        saveRootInteractionAndExecute(
                                                            listener,
                                                            memory,
                                                            inputDataSet,
                                                            mlTask,
                                                            isAsync,
                                                            outputs,
                                                            modelTensors,
                                                            mlAgent
                                                        );
                                                    }
                                                }, ex -> {
                                                    log.error("Failed to read conversation memory", ex);
                                                    listener.onFailure(ex);
                                                }));
                                        } else {
                                            executeAgent(inputDataSet, mlTask, isAsync, memoryId, mlAgent, outputs, modelTensors, listener);
                                        }
                                    } catch (Exception e) {
                                        log.error("Failed to parse ml agent {}", agentId, e);
                                        listener.onFailure(e);
                                    }
                                } else {
                                    listener
                                        .onFailure(
                                            new OpenSearchStatusException(
                                                "Failed to find agent with the provided agent id: " + agentId,
                                                RestStatus.NOT_FOUND
                                            )
                                        );
                                }
                            } catch (Exception e) {
                                log.error("Failed to get agent", e);
                                listener.onFailure(e);
                            }
                        }
                    });
            }
        } else {
            listener.onFailure(new ResourceNotFoundException("Agent index not found"));
        }
    }

    /**
     * save root interaction and start execute the agent
     * @param listener callback listener
     * @param memory memory instance
     * @param inputDataSet input
     * @param mlAgent agent to run
     */
    private void saveRootInteractionAndExecute(
        ActionListener<Output> listener,
        ConversationIndexMemory memory,
        RemoteInferenceInputDataSet inputDataSet,
        MLTask mlTask,
        boolean isAsync,
        List<ModelTensors> outputs,
        List<ModelTensor> modelTensors,
        MLAgent mlAgent
    ) {
        String appType = mlAgent.getAppType();
        String question = inputDataSet.getParameters().get(QUESTION);
        String regenerateInteractionId = inputDataSet.getParameters().get(REGENERATE_INTERACTION_ID);
        // Create root interaction ID
        ConversationIndexMessage msg = ConversationIndexMessage
            .conversationIndexMessageBuilder()
            .type(appType)
            .question(question)
            .response("")
            .finalAnswer(true)
            .sessionId(memory.getConversationId())
            .build();
        memory.save(msg, null, null, null, ActionListener.<CreateInteractionResponse>wrap(interaction -> {
            log.info("Created parent interaction ID: {}", interaction.getId());
            inputDataSet.getParameters().put(PARENT_INTERACTION_ID, interaction.getId());
            // only delete previous interaction when new interaction created
            if (regenerateInteractionId != null) {
                memory
                    .getMemoryManager()
                    .deleteInteractionAndTrace(
                        regenerateInteractionId,
                        ActionListener
                            .wrap(
                                deleted -> executeAgent(
                                    inputDataSet,
                                    mlTask,
                                    isAsync,
                                    memory.getConversationId(),
                                    mlAgent,
                                    outputs,
                                    modelTensors,
                                    listener
                                ),
                                e -> {
                                    log.error("Failed to regenerate for interaction {}", regenerateInteractionId, e);
                                    listener.onFailure(e);
                                }
                            )
                    );
            } else {
                executeAgent(inputDataSet, mlTask, isAsync, memory.getConversationId(), mlAgent, outputs, modelTensors, listener);
            }
        }, ex -> {
            log.error("Failed to create parent interaction", ex);
            listener.onFailure(ex);
        }));
    }

    /**
     * Executes the ML agent with the provided parameters.
     * Handles both synchronous and asynchronous execution modes.
     * 
     * @param inputDataSet The input data for the agent
     * @param mlTask The ML task associated with this execution
     * @param isAsync Whether to execute asynchronously
     * @param memoryId The memory ID for conversation context
     * @param mlAgent The ML agent to execute
     * @param outputs List to collect model tensor outputs
     * @param modelTensors List to collect individual model tensors
     * @param listener Callback listener for execution results
     */
    private void executeAgent(
        RemoteInferenceInputDataSet inputDataSet,
        MLTask mlTask,
        boolean isAsync,
        String memoryId,
        MLAgent mlAgent,
        List<ModelTensors> outputs,
        List<ModelTensor> modelTensors,
        ActionListener<Output> listener
    ) {
        String mcpConnectorConfigJSON = (mlAgent.getParameters() != null) ? mlAgent.getParameters().get(MCP_CONNECTORS_FIELD) : null;
        if (mcpConnectorConfigJSON != null && !mcpConnectorIsEnabled) {
            // MCP connector provided as tools but MCP feature is disabled, so abort.
            listener.onFailure(new OpenSearchException(ML_COMMONS_MCP_CONNECTOR_DISABLED_MESSAGE));
            return;
        }
        MLAgentRunner mlAgentRunner = getAgentRunner(mlAgent);
        // If async is true, index ML task and return the taskID. Also add memoryID to the task if it exists
        if (isAsync) {
            Map<String, Object> agentResponse = new HashMap<>();
            if (memoryId != null && !memoryId.isEmpty()) {
                agentResponse.put(MEMORY_ID, memoryId);
            }

            String parentInteractionId = inputDataSet.getParameters().get(PARENT_INTERACTION_ID);
            if (parentInteractionId != null && !parentInteractionId.isEmpty()) {
                agentResponse.put(PARENT_INTERACTION_ID, parentInteractionId);
            }
            mlTask.setResponse(agentResponse);
            mlTask.setAsync(true);

            indexMLTask(mlTask, ActionListener.wrap(indexResponse -> {
                String taskId = indexResponse.getId();
                mlTask.setTaskId(taskId);

                MLTaskOutput outputBuilder = MLTaskOutput.builder().taskId(taskId).status(MLTaskState.RUNNING.toString()).build();

                if (memoryId != null && !memoryId.isEmpty()) {
                    outputBuilder.setResponse(agentResponse);
                }
                listener.onResponse(outputBuilder);
                ActionListener<Object> agentActionListener = createAsyncTaskUpdater(mlTask, outputs, modelTensors);
                inputDataSet.getParameters().put(TASK_ID_FIELD, taskId);
                mlAgentRunner.run(mlAgent, inputDataSet.getParameters(), agentActionListener);
            }, e -> {
                log.error("Failed to create task for agent async execution", e);
                listener.onFailure(e);
            }));
        } else {
            ActionListener<Object> agentActionListener = createAgentActionListener(listener, outputs, modelTensors, mlAgent.getType());
            mlAgentRunner.run(mlAgent, inputDataSet.getParameters(), agentActionListener);
        }
    }

    @SuppressWarnings("removal")
    private ActionListener<Object> createAgentActionListener(
        ActionListener<Output> listener,
        List<ModelTensors> outputs,
        List<ModelTensor> modelTensors,
        String agentType
    ) {
        return ActionListener.wrap(output -> {
            if (output != null) {
                processOutput(output, modelTensors);
                listener.onResponse(ModelTensorOutput.builder().mlModelOutputs(outputs).build());
            } else {
                listener.onResponse(null);
            }
        }, ex -> {
            log.error("Failed to run {} agent", agentType, ex);
            listener.onFailure(ex);
        });
    }

    private ActionListener<Object> createAsyncTaskUpdater(MLTask mlTask, List<ModelTensors> outputs, List<ModelTensor> modelTensors) {
        String taskId = mlTask.getTaskId();
        Map<String, Object> agentResponse = new HashMap<>();
        Map<String, Object> updatedTask = new HashMap<>();

        return ActionListener.wrap(output -> {
            if (output != null) {
                processOutput(output, modelTensors);
                agentResponse.put(INFERENCE_RESULT_FIELD, outputs);
            } else {
                agentResponse.put(ERROR_MESSAGE, "No output found from agent execution");
            }

            mlTask.setResponse(agentResponse);
            updatedTask.put(RESPONSE_FIELD, agentResponse);
            updatedTask.put(STATE_FIELD, MLTaskState.COMPLETED);
            updateMLTaskDirectly(
                taskId,
                updatedTask,
                client,
                ActionListener
                    .wrap(
                        response -> log.info("Updated ML task {} with agent execution results", taskId),
                        e -> log.error("Failed to update ML task {} with agent execution results", taskId)
                    )
            );
        }, ex -> {
            agentResponse.put(ERROR_MESSAGE, ex.getMessage());

            updatedTask.put(RESPONSE_FIELD, agentResponse);
            updatedTask.put(STATE_FIELD, MLTaskState.FAILED);
            mlTask.setResponse(agentResponse);

            updateMLTaskDirectly(
                taskId,
                updatedTask,
                client,
                ActionListener
                    .wrap(
                        response -> log.info("Updated ML task {} with agent execution failed reason", taskId),
                        e -> log.error("Failed to update ML task {} with agent execution results", taskId)
                    )
            );
        });
    }

    /**
     * Factory method to create the appropriate MLAgentRunner based on agent type.
     * Supports FLOW, CONVERSATIONAL_FLOW, CONVERSATIONAL, and PLAN_EXECUTE_AND_REFLECT agent types.
     * 
     * @param mlAgent The ML agent configuration
     * @return The appropriate MLAgentRunner instance
     * @throws IllegalArgumentException if the agent type is not supported
     */
    @VisibleForTesting
    protected MLAgentRunner getAgentRunner(MLAgent mlAgent) {
        final MLAgentType agentType = MLAgentType.from(mlAgent.getType().toUpperCase(Locale.ROOT));

        switch (agentType) {
            case FLOW:
                return createFlowAgentRunner();
            case CONVERSATIONAL_FLOW:
                return createConversationalFlowAgentRunner();
            case CONVERSATIONAL:
                return createChatAgentRunner();
            case PLAN_EXECUTE_AND_REFLECT:
                return createPlanExecuteAndReflectAgentRunner();
            default:
                throw new IllegalArgumentException("Unsupported agent type: " + mlAgent.getType());
        }
    }

    /**
     * Creates a flow agent runner with common dependencies
     */
    private MLFlowAgentRunner createFlowAgentRunner() {
        return new MLFlowAgentRunner(
            client,
            settings,
            clusterService,
            xContentRegistry,
            toolFactories,
            memoryFactoryMap,
            sdkClient,
            encryptor
        );
    }

    /**
     * Creates a conversational flow agent runner with common dependencies
     */
    private MLConversationalFlowAgentRunner createConversationalFlowAgentRunner() {
        return new MLConversationalFlowAgentRunner(
            client,
            settings,
            clusterService,
            xContentRegistry,
            toolFactories,
            memoryFactoryMap,
            sdkClient,
            encryptor
        );
    }

    /**
     * Creates a chat agent runner with common dependencies
     */
    private MLChatAgentRunner createChatAgentRunner() {
        return new MLChatAgentRunner(
            client,
            settings,
            clusterService,
            xContentRegistry,
            toolFactories,
            memoryFactoryMap,
            sdkClient,
            encryptor
        );
    }

    /**
     * Creates a plan-execute-and-reflect agent runner with common dependencies
     */
    private MLPlanExecuteAndReflectAgentRunner createPlanExecuteAndReflectAgentRunner() {
        return new MLPlanExecuteAndReflectAgentRunner(
            client,
            settings,
            clusterService,
            xContentRegistry,
            toolFactories,
            memoryFactoryMap,
            sdkClient,
            encryptor
        );
    }

    /**
     * Processes various types of agent output and converts them to ModelTensor format.
     * Handles ModelTensorOutput, ModelTensor, Lists, and generic objects.
     * 
     * @param output The output from agent execution
     * @param modelTensors List to collect processed model tensors
     * @throws PrivilegedActionException if JSON serialization fails
     */
    @SuppressWarnings("removal")
    public void processOutput(Object output, List<ModelTensor> modelTensors) throws PrivilegedActionException {
        if (output == null) {
            return;
        }

        if (output instanceof ModelTensorOutput) {
            processModelTensorOutput((ModelTensorOutput) output, modelTensors);
        } else if (output instanceof ModelTensor) {
            modelTensors.add((ModelTensor) output);
        } else if (output instanceof List) {
            processListOutput((List<?>) output, modelTensors);
        } else {
            processGenericOutput(output, modelTensors);
        }
    }

    /**
     * Processes ModelTensorOutput by extracting all model tensors
     */
    private void processModelTensorOutput(ModelTensorOutput modelTensorOutput, List<ModelTensor> modelTensors) {
        modelTensorOutput.getMlModelOutputs().forEach(outs -> { modelTensors.addAll(outs.getMlModelTensors()); });
    }

    /**
     * Processes List output by determining the type of list elements
     */
    @SuppressWarnings("removal")
    private void processListOutput(List<?> outputList, List<ModelTensor> modelTensors) throws PrivilegedActionException {
        if (outputList.isEmpty()) {
            return;
        }

        Object firstElement = outputList.get(0);
        if (firstElement instanceof ModelTensor) {
            modelTensors.addAll((List<ModelTensor>) outputList);
        } else if (firstElement instanceof ModelTensors) {
            ((List<ModelTensors>) outputList).forEach(outs -> { modelTensors.addAll(outs.getMlModelTensors()); });
        } else {
            // Convert list to JSON string
            String result = AccessController.doPrivileged((PrivilegedExceptionAction<String>) () -> {
                Gson gson = new Gson();
                return gson.toJson(outputList);
            });
            modelTensors.add(ModelTensor.builder().name("response").result(result).build());
        }
    }

    /**
     * Processes generic output by converting to string or JSON
     */
    @SuppressWarnings("removal")
    private void processGenericOutput(Object output, List<ModelTensor> modelTensors) throws PrivilegedActionException {
        String result;
        if (output instanceof String) {
            result = (String) output;
        } else {
            result = AccessController.doPrivileged((PrivilegedExceptionAction<String>) () -> {
                Gson gson = new Gson();
                return gson.toJson(output);
            });
        }
        modelTensors.add(ModelTensor.builder().name("response").result(result).build());
    }

    public void indexMLTask(MLTask mlTask, ActionListener<IndexResponse> listener) {
        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            sdkClient
                .putDataObjectAsync(
                    PutDataObjectRequest.builder().index(ML_TASK_INDEX).tenantId(mlTask.getTenantId()).dataObject(mlTask).build()
                )
                .whenComplete((r, throwable) -> {
                    context.restore();
                    if (throwable != null) {
                        Exception cause = SdkClientUtils.unwrapAndConvertToException(throwable);
                        log.error("Failed to index ML task", cause);
                        listener.onFailure(cause);
                    } else {
                        try {
                            IndexResponse indexResponse = IndexResponse.fromXContent(r.parser());
                            log.info("Task creation result: {}, Task id: {}", indexResponse.getResult(), indexResponse.getId());
                            listener.onResponse(indexResponse);
                        } catch (Exception e) {
                            listener.onFailure(e);
                        }
                    }
                });
        } catch (Exception e) {
            log.error("Failed to create ML task for {}, {}", mlTask.getFunctionName(), mlTask.getTaskType(), e);
            listener.onFailure(e);
        }
    }

    /**
     * Inner class to encapsulate execution context and reduce parameter passing
     */
    private static class ExecutionContext {
        private final String agentId;
        private final String tenantId;
        private final Boolean isAsync;
        private final RemoteInferenceInputDataSet inputDataSet;
        private final List<ModelTensors> outputs;
        private final List<ModelTensor> modelTensors;

        public ExecutionContext(
            String agentId,
            String tenantId,
            Boolean isAsync,
            RemoteInferenceInputDataSet inputDataSet,
            List<ModelTensors> outputs,
            List<ModelTensor> modelTensors
        ) {
            this.agentId = agentId;
            this.tenantId = tenantId;
            this.isAsync = isAsync;
            this.inputDataSet = inputDataSet;
            this.outputs = outputs;
            this.modelTensors = modelTensors;
        }

        public String getAgentId() {
            return agentId;
        }

        public String getTenantId() {
            return tenantId;
        }

        public Boolean getIsAsync() {
            return isAsync;
        }

        public RemoteInferenceInputDataSet getInputDataSet() {
            return inputDataSet;
        }

        public List<ModelTensors> getOutputs() {
            return outputs;
        }

        public List<ModelTensor> getModelTensors() {
            return modelTensors;
        }
    }
}
