/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */

package com.huawei.hms.mlsdk.sample.modelinterpreter;

import android.content.Context;
import android.graphics.Bitmap;

import com.huawei.hms.mlsdk.custom.MLModelOutputs;
import com.huawei.hms.mlsdk.sample.modelinterpreter.model.ImageLableModel;
import com.huawei.hms.mlsdk.sample.modelinterpreter.utils.LabelUtils;

import java.util.ArrayList;
import java.util.List;

public abstract class ModelOperator {
    public enum Model {
        LABEL
    }

    protected String modelName;

    protected String modelFullName;

    protected String modelLabelFile;

    protected int batchNum = 0;

    protected abstract int getInputType();

    protected abstract int getOutputType();

    protected abstract Object getInput(Bitmap bmp);

    protected abstract int[] getInputShape();

    protected abstract ArrayList<int[]> getOutputShapeList();

    protected abstract String resultPostProcess(MLModelOutputs output);

    public static ModelOperator create(Context activity, Model model) {
        if (model == Model.LABEL) {
            return new ImageLableModel(activity);
        } else {
            throw new UnsupportedOperationException();
        }
    }

    protected String getExecutorResult(List<String> label, float[] result) {
        return LabelUtils.processResult(label, result);
    }

    protected String getModelName() {
        return modelName;
    }

    protected String getModelFullName() {
        return modelFullName;
    }
}
