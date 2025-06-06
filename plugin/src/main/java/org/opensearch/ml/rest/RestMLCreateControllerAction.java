/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
import static org.opensearch.ml.utils.MLExceptionUtils.CONTROLLER_DISABLED_ERR_MSG;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_MODEL_ID;
import static org.opensearch.ml.utils.RestActionUtils.getParameterId;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import org.opensearch.OpenSearchParseException;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.controller.MLController;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.controller.MLCreateControllerAction;
import org.opensearch.ml.common.transport.controller.MLCreateControllerRequest;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.transport.client.node.NodeClient;

import com.google.common.collect.ImmutableList;

public class RestMLCreateControllerAction extends BaseRestHandler {

    public final static String ML_CREATE_CONTROLLER_ACTION = "ml_create_controller_action";
    private final MLFeatureEnabledSetting mlFeatureEnabledSetting;

    /**
     * Constructor
     */
    public RestMLCreateControllerAction(MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
    }

    @Override
    public String getName() {
        return ML_CREATE_CONTROLLER_ACTION;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList
            .of(new Route(RestRequest.Method.POST, String.format(Locale.ROOT, "%s/controllers/{%s}", ML_BASE_URI, PARAMETER_MODEL_ID)));
    }

    @Override
    public RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        MLCreateControllerRequest createControllerRequest = getRequest(request);
        return channel -> {
            client.execute(MLCreateControllerAction.INSTANCE, createControllerRequest, new RestToXContentListener<>(channel));
        };
    }

    /**
     * Creates a MLCreateControllerRequest from a RestRequest
     *
     * @param request RestRequest
     * @return MLCreateControllerRequest
     */
    private MLCreateControllerRequest getRequest(RestRequest request) throws IOException {
        if (!mlFeatureEnabledSetting.isControllerEnabled()) {
            throw new IllegalStateException(CONTROLLER_DISABLED_ERR_MSG);
        }

        if (!request.hasContent()) {
            throw new OpenSearchParseException("Create model controller request has empty body");
        }
        // Model ID can only be set here.
        String modelId = getParameterId(request, PARAMETER_MODEL_ID);
        XContentParser parser = request.contentParser();
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
        MLController controllerInput = MLController.parse(parser);
        controllerInput.setModelId(modelId);
        return new MLCreateControllerRequest(controllerInput);
    }
}
