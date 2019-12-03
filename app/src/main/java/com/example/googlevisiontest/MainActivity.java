package com.example.googlevisiontest;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.util.SparseArray;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;
import com.google.android.gms.vision.face.Landmark;

import android.net.Uri;
import android.content.ActivityNotFoundException;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.model.Model;
import android.app.Activity;


import android.os.Bundle;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    Button btnUpload;
    Button btnProgress;
    TextView textView;

    private static final int IMAGE_PICK_CODE = 1000;
    private static final int PERMISSION_CODE = 1001;

    Bitmap eyePatchBitmap;
    Bitmap flowerLine;
    Canvas canvas;

    Paint rectPaint = new Paint();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = (ImageView)findViewById(R.id.imageView);
        btnUpload = (Button) findViewById(R.id.btnUpload);
        btnProgress = (Button)findViewById(R.id.btnProgress);
        textView = (TextView) findViewById(R.id.textView);

        btnUpload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
                    if(checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED){
                        // permision not granted, request it
                        String [] permissions = {Manifest.permission.READ_EXTERNAL_STORAGE};
                        // show popup for runtime permission
                        requestPermissions(permissions,PERMISSION_CODE);

                    }
                    else{
                        // permission already granted
                        pickImageFromGallery();

                    }
                }
                else{
                    // system os is less then marshmallow
                    pickImageFromGallery();
                }
            }
        });



        btnProgress.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final Bitmap myBitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
                imageView.setImageBitmap(myBitmap);

                rectPaint.setStrokeWidth(5);
                rectPaint.setColor(Color.WHITE);
                rectPaint.setStyle(Paint.Style.STROKE);

                final Bitmap tempBitmap = Bitmap.createBitmap(myBitmap.getWidth(),myBitmap.getHeight(), Bitmap.Config.RGB_565);
                canvas  = new Canvas(tempBitmap);
                canvas.drawBitmap(myBitmap,0,0,null);
                FaceDetector faceDetector = new FaceDetector.Builder(getApplicationContext())
                        .setTrackingEnabled(false)
                        .setLandmarkType(FaceDetector.ALL_LANDMARKS)
                        .setMode(FaceDetector.FAST_MODE)
                        .build();

                if(!faceDetector.isOperational())
                {
                    Toast.makeText(MainActivity.this, "Face Detector could not be set up on your device", Toast.LENGTH_SHORT).show();
                    return;
                }
                Frame frame = new Frame.Builder().setBitmap(myBitmap).build();
                SparseArray<Face> sparseArray = faceDetector.detect(frame);

                Face face = sparseArray.valueAt(0);
                int x1=(int)face.getPosition().x-400;
                int y1 =(int)face.getPosition().y-400;
                int x2 = (int) (x1+face.getWidth()+800);
                int y2=(int)(y1+face.getHeight()+800);
                RectF rectF = new RectF(x1,y1,x2,y2);
                canvas.drawRoundRect(rectF,2,2,rectPaint);

                if(x1>=0 && y1>=0 && x2-x1!=0 && y2-y1!=0){
                    Bitmap cropedBitmap = Bitmap.createBitmap(myBitmap,x1,y1,x2-x1,y2-y1);
                    imageView.setImageBitmap(cropedBitmap);
                    imageProcessing(cropedBitmap);
                }else{
                    imageProcessing(myBitmap);
                }

            }
        });
    }
    private MappedByteBuffer loadModelFile(Activity activity,String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    private  void imageProcessing(Bitmap bitmap){
        // Initialization code
        // Create an ImageProcessor with all ops required. For more ops, please
        // refer to the ImageProcessor Architecture section in this README.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(257, 257, ResizeOp.ResizeMethod.BILINEAR))
                        .build();

        // Create a TensorImage object, this creates the tensor the TensorFlow Lite
        // interpreter needs
        TensorImage tImage = new TensorImage(DataType.FLOAT32);

        // Analysis code for every frame
        // Preprocess the image
        tImage.load(bitmap);
        tImage = imageProcessor.process(tImage);


        // Create a container for the result and specify that this is a quantized model.
        // Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
        TensorBuffer probabilityBuffer =
                TensorBuffer.createFixedSize(new int[]{1, 1}, DataType.FLOAT32);

        // Initialise the model
        Interpreter tflite =null;
        try{
            tflite = new Interpreter(loadModelFile(MainActivity.this,"genderModel.tflite"));
        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);

        }

        // Running inference
        if( tflite != null) {
            tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer().rewind());
            textView.setText(probabilityBuffer.getFloatArray()[0]+"");
        }



    }

    private void pickImageFromGallery(){
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, IMAGE_PICK_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch(requestCode){
            case PERMISSION_CODE:{
                if(grantResults.length > 0 && grantResults[0]==PackageManager.PERMISSION_GRANTED){
                    pickImageFromGallery();
                }
                else{
                    //permission denied
                    Toast.makeText(this,"Permission denied!",Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode,resultCode,data);
        if(resultCode == RESULT_OK && requestCode == IMAGE_PICK_CODE){
            //set image to image view
            imageView.setImageURI(data.getData());
        }


    }


}

