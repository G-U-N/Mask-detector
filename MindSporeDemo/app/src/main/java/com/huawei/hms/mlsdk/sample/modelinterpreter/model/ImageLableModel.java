/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */

package com.huawei.hms.mlsdk.sample.modelinterpreter.model;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;

import com.huawei.hms.mlsdk.custom.MLModelDataType;
import com.huawei.hms.mlsdk.custom.MLModelOutputs;
import com.huawei.hms.mlsdk.sample.modelinterpreter.ModelOperator;
import com.huawei.hms.mlsdk.sample.modelinterpreter.utils.LabelUtils;

import java.util.ArrayList;
import java.util.List;

public class ImageLableModel extends ModelOperator {
    private static final int BITMAP_SIZE = 224;
    private static final float[] IMAGE_MEAN = new float[] {0.485f * 255f, 0.456f * 255f, 0.406f * 255f};
    private static final float[] IMAGE_STD = new float[] {0.229f * 255f, 0.224f * 255f, 0.225f * 255f};
    private int outputSize;
    private List<String> labelList;
    private Context mContext;

    public ImageLableModel(Context context) {
        mContext = context;
        modelName = "mindspore";
        modelFullName = "mindspore" + ".ms";
        modelLabelFile = "labels.txt";
        labelList = LabelUtils.readLabels(mContext, modelLabelFile);
    }

    @Override
    protected int getInputType() {
        return MLModelDataType.FLOAT32;
    }

    @Override
    protected int getOutputType() {
        return MLModelDataType.FLOAT32;
    }

    @Override
    protected Object getInput(Bitmap inputBitmap) {
        final float[][][][] input = new float[1][BITMAP_SIZE][BITMAP_SIZE][3];
        for (int h = 0; h < BITMAP_SIZE; h++) {
            for (int w = 0; w < BITMAP_SIZE; w++) {
                int pixel = inputBitmap.getPixel(w, h);
                input[batchNum][h][w][0] = ((Color.red(pixel) - IMAGE_MEAN[0])) / IMAGE_STD[0];
                input[batchNum][h][w][1] = ((Color.green(pixel) - IMAGE_MEAN[1])) / IMAGE_STD[1];
                input[batchNum][h][w][2] = ((Color.blue(pixel) - IMAGE_MEAN[2])) / IMAGE_STD[2];
            }
        }
        return input;
    }

    @Override
    protected int[] getInputShape() {
        return new int[] {1, BITMAP_SIZE, BITMAP_SIZE, 3};
    }

    @Override
    protected ArrayList<int[]> getOutputShapeList() {
        ArrayList<int[]> outputShapeList = new ArrayList<>();
        int[] outputShape = new int[] {1, labelList.size()};
        outputShapeList.add(outputShape);
        return outputShapeList;
    }

    @Override
    protected String resultPostProcess(MLModelOutputs output) {
        float[][] result = output.getOutput(0);
        float[] probabilities = result[0];
        return getExecutorResult(labelList, probabilities);
    }
}
