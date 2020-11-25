package com.huawei.hms.mlsdk.sample.modelinterpreter;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.widget.Toast;

import com.huawei.hmf.tasks.OnCompleteListener;
import com.huawei.hmf.tasks.OnFailureListener;
import com.huawei.hmf.tasks.OnSuccessListener;
import com.huawei.hmf.tasks.Task;
import com.huawei.hms.mlsdk.common.MLException;
import com.huawei.hms.mlsdk.custom.MLCustomLocalModel;
import com.huawei.hms.mlsdk.custom.MLModelExecutor;
import com.huawei.hms.mlsdk.custom.MLModelExecutorSettings;
import com.huawei.hms.mlsdk.custom.MLModelInputOutputSettings;
import com.huawei.hms.mlsdk.custom.MLModelInputs;
import com.huawei.hms.mlsdk.custom.MLModelOutputs;
import com.huawei.hms.mlsdk.sample.modelinterpreter.utils.CropBitMap;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.ArrayList;

public class InterpreterManager {
    private static final String TAG = "CustModelActivity";
    private static final int BITMAP_WIDTH = 224; // 128, 224
    private static final int BITMAP_HEIGHT = 224; // 128, 224

    private WeakReference<Context> weakContext;
    private MLModelExecutor modelExecutor;
    private ModelOperator.Model modelType;
    private ModelOperator mModelOperator;

    private String mModelName;
    private String mModelFullName; // .om, .mslite, .ms

    public InterpreterManager(Context context, ModelOperator.Model modelType) {
        this.modelType = modelType;
        weakContext = new WeakReference<>(context);

        initEnvironment();
    }

    private void initEnvironment() {
        mModelOperator = ModelOperator.create(weakContext.get(), modelType);
        mModelName = mModelOperator.getModelName();
        mModelFullName = mModelOperator.getModelFullName();
    }

    public void asset(Bitmap bitmap) {
        if (dumpBitmapInfo(bitmap)) {
            return;
        }

        MLCustomLocalModel localModel =
            new MLCustomLocalModel.Factory(mModelName).setAssetPathFile(mModelFullName).create();
        MLModelExecutorSettings settings = new MLModelExecutorSettings.Factory(localModel).create();

        try {
            modelExecutor = MLModelExecutor.getInstance(settings);
            executorImpl(modelExecutor, bitmap);
        } catch (MLException error) {
            error.printStackTrace();
        }
    }

    private void showToast(final String text) {
        Toast toast = Toast.makeText(weakContext.get(), text, Toast.LENGTH_SHORT);
        toast.show();
    }

    private boolean dumpBitmapInfo(Bitmap bitmap) {
        if (bitmap == null) {
            return true;
        }
        final int width = bitmap.getWidth();
        final int height = bitmap.getHeight();
        Log.e(TAG, "bitmap width is " + width + " height " + height);
        return false;
    }

    private Bitmap processBitMap(Bitmap bitmap) {
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        CropBitMap crop = new CropBitMap(cropSize, cropSize);
        final Bitmap cropBitmap = crop.getCropBitmap(bitmap);
        dumpBitmapInfo(cropBitmap);
        return Bitmap.createScaledBitmap(cropBitmap, BITMAP_WIDTH, BITMAP_HEIGHT, false);
    }

    private void executorImpl(final MLModelExecutor executor, Bitmap bitmap) {
        Bitmap inputBitmap = processBitMap(bitmap);
        Object input = mModelOperator.getInput(inputBitmap);
        Log.d(TAG, "interpret pre process");

        MLModelInputs inputs = null;

        try {
            inputs = new MLModelInputs.Factory().add(input).create();
        } catch (MLException e) {
            Log.e(TAG, "add inputs failed! " + e.getMessage());
        }

        MLModelInputOutputSettings inOutSettings = null;
        try {
            MLModelInputOutputSettings.Factory settingsFactory = new MLModelInputOutputSettings.Factory();
            settingsFactory.setInputFormat(0, mModelOperator.getInputType(), mModelOperator.getInputShape());
            ArrayList<int[]> outputSettingsList = mModelOperator.getOutputShapeList();
            for (int i = 0; i < outputSettingsList.size(); i++) {
                settingsFactory.setOutputFormat(i, mModelOperator.getOutputType(), outputSettingsList.get(i));
            }
            inOutSettings = settingsFactory.create();
        } catch (MLException e) {
            Log.e(TAG, "set input output format failed! " + e.getMessage());
        }

        Log.d(TAG, "interpret start");
        execModel(inputs, inOutSettings);
    }

    private void execModel(MLModelInputs inputs, MLModelInputOutputSettings outputSettings) {
        modelExecutor.exec(inputs, outputSettings).addOnSuccessListener(new OnSuccessListener<MLModelOutputs>() {
            @Override
            public void onSuccess(MLModelOutputs mlModelOutputs) {
                Log.i(TAG, "interpret get result");
                String result = mModelOperator.resultPostProcess(mlModelOutputs);
                showResult(result);
                showToast("success");
                Log.i(TAG, "result: " + result);
            }
        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(Exception e) {
                e.printStackTrace();
                Log.e(TAG, "interpret failed, because " + e.getMessage());
                showToast("failed");
            }
        }).addOnCompleteListener(new OnCompleteListener<MLModelOutputs>() {
            @Override
            public void onComplete(Task<MLModelOutputs> task) {
                try {
                    modelExecutor.close();
                } catch (IOException error) {
                    error.printStackTrace();
                }
            }
        });
    }

    private void showResult(final String result) {
        final CustModelActivity activity = (CustModelActivity) weakContext.get();
        activity.resultText.post(new Runnable() {
            @Override
            public void run() {
                activity.resultText.setText(result);
            }
        });
    }

}
