/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.example.inferentia.embeddings;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import com.example.inferentia.InferentiaDemo;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;

public class InferentiaSentenceEmbeddingDemo {

    public static void main(String[] args) throws IOException, ModelException, TranslateException, URISyntaxException {
        String engineName = "PyTorch";
        String version = Engine.getEngine(engineName).getVersion();
        System.out.println("Running inference with PyTorch: " + version);


        Device[] devices = Engine.getEngine(engineName).getDevices();
        System.out.println("Find " + devices.length + " devices");
        for (Device device : devices) {
            System.out.println(device.getDeviceId() + ", " + device.getDeviceType() + ", " + device.isGpu());
        }
        URL resource = InferentiaDemo.class.getResource("/all-MiniLM-L6-v2_torchscript_sentence-transformer.zip");
        Path modelPath = Path.of(resource.toURI());
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelPath)
//                        .optTranslatorFactory(new HuggingfaceTextEmbeddingTranslatorFactory())
                        .optTranslator(new SentenceTransformerTextEmbeddingTranslator())
                        .build();

        try (ZooModel<Input, Output> model = ModelZoo.loadModel(criteria);
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add("this is a test sentence");
            Output result = predictor.predict(input);
            System.out.println(result);
        }
    }

}
