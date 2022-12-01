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
package com.example.inferentia;

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
import com.example.inferentia.embeddings.HuggingfaceTextEmbeddingTranslatorFactory;
import com.example.inferentia.embeddings.SentenceTransformerTextEmbeddingTranslator;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

public class InferentiaDemo {
    public static void main(String[] args) throws IOException, ModelException, TranslateException, URISyntaxException {
        String engineName = "PyTorch";
        String version = Engine.getEngine(engineName).getVersion();
        System.out.println("Running inference with PyTorch: " + version);


        Device[] devices = Engine.getEngine(engineName).getDevices();
        System.out.println("Find " + devices.length + " devices");
        for (Device device : devices) {
            System.out.println(device.getDeviceId() + ", " + device.getDeviceType() + ", " + device.isGpu());
        }
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get("models/allmini_traced_inf1"))
                        .optTranslatorFactory(new HuggingfaceTextEmbeddingTranslatorFactory())
//                        .optTranslator(new SentenceTransformerTextEmbeddingTranslator())
                        .build();

        try (ZooModel<Input, Output> model = ModelZoo.loadModel(criteria);
             Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add("this is a test sentence");
            Output result = predictor.predict(input);
            System.out.println(result);
        }
    }
//    public static void main(String[] args) throws IOException, ModelException, TranslateException {
//        String version = Engine.getEngine("PyTorch").getVersion();
//        System.out.println("Running inference with PyTorch: " + version);
//
//        // You need manually load libneuron_op.so prior to 0.12.0, uncomment the following code
//        // if use 0.11.0 release. And libneuron_op.so must be loaded after PyTorch engine is loaded.
//        /*
//        String extraPath = System.getenv("PYTORCH_EXTRA_LIBRARY_PATH");
//        if (extraPath != null) {
//            System.load(extraPath);
//        } else {
//            System.loadLibrary("neuron_op");
//        }
//        */
//
//        String url = "https://resources.djl.ai/images/kitten.jpg";
//        Image img = ImageFactory.getInstance().fromUrl(url);
//        Criteria<Image, Classifications> criteria =
//                Criteria.builder()
//                        .setTypes(Image.class, Classifications.class)
//                        .optModelPath(Paths.get("models/inferentia/resnet50"))
//                        .optTranslator(getTranslator())
//                        .build();
//
//        try (ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
//                Predictor<Image, Classifications> predictor = model.newPredictor()) {
//            Classifications result = predictor.predict(img);
//            System.out.println(result);
//        }
//    }
//
//    private static Translator<Image, Classifications> getTranslator() {
//        return ImageClassificationTranslator.builder()
//                .addTransform(new CenterCrop())
//                .addTransform(new Resize(224, 224))
//                .addTransform(new ToTensor())
//                .optSynsetUrl(InferentiaDemo.class.getResource("/synset.txt").toString())
//                .optApplySoftmax(true)
//                .build();
//    }
}
