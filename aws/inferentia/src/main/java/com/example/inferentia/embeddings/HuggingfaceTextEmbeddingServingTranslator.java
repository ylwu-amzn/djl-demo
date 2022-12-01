/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package com.example.inferentia.embeddings;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import com.google.gson.Gson;


public class HuggingfaceTextEmbeddingServingTranslator implements Translator<Input, Output> {

    private Translator<String, float[]> translator;
    private Gson gson;

    public HuggingfaceTextEmbeddingServingTranslator(Translator<String, float[]> translator) {
        this.translator = translator;
    }

    @Override
    public Batchifier getBatchifier() {
        return translator.getBatchifier();
    }

    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        translator.prepare(ctx);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NDList processInput(TranslatorContext ctx, Input input) throws Exception {
        String text = input.getData().getAsString();
        return translator.processInput(ctx, text);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) throws Exception {
        float[] ret = translator.processOutput(ctx, list);

        Output output = new Output();
        output.add(gson.toJson(ret));
        return output;
    }

}