
package com.huawei.hms.mlsdk.sample.modelinterpreter;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.huawei.hms.mlsdk.common.MLApplication;
import com.huawei.hms.mlsdk.sample.modelinterpreter.utils.FileUtil;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;

public class CustModelActivity extends AppCompatActivity {
    private static final String TAG = "CustModelActivity";

    private InterpreterManager interpreterManager;

    private static final int REQUEST_PERMISSION_CODE = 10;

    private static final int RC_CHOOSE_PHOTO = 2;

    private Bitmap bitmap;

    @BindView(R.id.Asset)
    public Button assetButton;

    @BindView(R.id.choosePicture)
    public Button choose;

    @BindView(R.id.capturedImageView)
    public ImageView capturedImage;

    @BindView(R.id.resultText)
    public TextView resultText;

    ModelOperator.Model modelType = ModelOperator.Model.LABEL;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_cust_model);
        ButterKnife.bind(this);

        interpreterManager = new InterpreterManager(this, modelType);
    }

    @Override
    public void onActivityResult(final int requestCode, final int resultCode, final Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        bitmap = processIntent(requestCode, resultCode, data);
        capturedImage.setImageBitmap(bitmap);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
        @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSION_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                choosePhoto();
            } else {
                Toast.makeText(getApplicationContext(), "no permission", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private Bitmap processIntent(final int requestCode, final int resultCode, final Intent data) {
        if (requestCode == RC_CHOOSE_PHOTO) {
            if (data == null) {
                return null;
            }
            Uri uri = data.getData();
            String filePath = FileUtil.getFilePathByUri(this, uri);

            if (!TextUtils.isEmpty(filePath)) {
                Log.e(TAG, "file is " + filePath);
                return BitmapFactory.decodeFile(filePath);
            }
        }
        return null;
    }

    private void checkPermissionIfNecessary() {
        if (ContextCompat.checkSelfPermission(getApplicationContext(),
            Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this,
                new String[] {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                REQUEST_PERMISSION_CODE);
        } else {
            Intent intentToPickPic = new Intent(Intent.ACTION_PICK, null);
            intentToPickPic.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
            startActivityForResult(intentToPickPic, RC_CHOOSE_PHOTO);
        }
    }

    @OnClick(R.id.choosePicture)
    public void choosePhoto() {
        checkPermissionIfNecessary();
    }

    @OnClick(R.id.Asset)
    public void assetOnclick() {
        interpreterManager.asset(bitmap);
    }

    private int index = 0;

    @OnClick(R.id.selectImageFromAsset)
    public void assetsDemoImage() {
        try {
            String[] files = getAssets().list("demoimages");
            Log.d("hiai", Arrays.toString(files));
            if(files != null && files.length > 0) {
                if (index >= files.length) {
                    index = index % files.length;
                }
                InputStream stream = getAssets().open("demoimages/" + files[index]);
                index++;
                bitmap = BitmapFactory.decodeStream(stream);
                capturedImage.setImageBitmap(bitmap);
            }
        } catch (IOException e) {
        }
    }
}