/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 *
 */

package org.opensearch.ml.common.parameter;

import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import org.opensearch.common.ParseField;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.annotation.MLAlgoParameter;

import java.io.IOException;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

@Data
@MLAlgoParameter(algorithms={FunctionName.KMEANS})
public class KMeansParams implements MLAlgoParams {

    public static final String PARSE_FIELD_NAME = FunctionName.KMEANS.name();
    public static final NamedXContentRegistry.Entry XCONTENT_REGISTRY = new NamedXContentRegistry.Entry(
            MLAlgoParams.class,
            new ParseField(PARSE_FIELD_NAME),
            it -> parse(it)
    );

    public static final String CENTROIDS_FIELD = "centroids";
    public static final String ITERATIONS_FIELD = "iterations";
    public static final String DISTANCE_TYPE_FIELD = "distance_type";

    //The number of centroids to use.
    private Integer centroids;
    //The maximum number of iterations
    private Integer iterations;
    //The distance function.
    private DistanceType distanceType;
    //TODO: expose number of thread and seed?

    @Builder
    public KMeansParams(Integer centroids, Integer iterations, DistanceType distanceType) {
        this.centroids = centroids;
        this.iterations = iterations;
        this.distanceType = distanceType;
    }

    public KMeansParams(StreamInput in) throws IOException {
        this.centroids = in.readOptionalInt();
        this.iterations = in.readOptionalInt();
        if (in.readBoolean()) {
            this.distanceType = in.readEnum(DistanceType.class);
        }
    }

    public static MLAlgoParams parse(XContentParser parser) throws IOException {
        Integer k = null;
        Integer iterations = null;
        DistanceType distanceType = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case CENTROIDS_FIELD:
                    k = parser.intValue();
                    break;
                case ITERATIONS_FIELD:
                    iterations = parser.intValue();
                    break;
                case DISTANCE_TYPE_FIELD:
                    distanceType = DistanceType.valueOf(parser.text());
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new KMeansParams(k, iterations, distanceType);
    }

    @Override
    public String getWriteableName() {
        return PARSE_FIELD_NAME;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeOptionalInt(centroids);
        out.writeOptionalInt(iterations);
        if (distanceType != null) {
            out.writeBoolean(true);
            out.writeEnum(distanceType);
        } else {
            out.writeBoolean(false);
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (centroids != null) {
            builder.field(CENTROIDS_FIELD, centroids);
        }
        if (iterations != null) {
            builder.field(ITERATIONS_FIELD, iterations);
        }
        if (distanceType != null) {
            builder.field(DISTANCE_TYPE_FIELD, distanceType.name());
        }
        builder.endObject();
        return builder;
    }

    @Override
    public int getVersion() {
        return 1;
    }

    public enum DistanceType {
        EUCLIDEAN,
        COSINE,
        L1
    }
}