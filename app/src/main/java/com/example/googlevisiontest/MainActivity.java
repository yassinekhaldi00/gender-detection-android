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
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
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
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    Button btnUpload;
    Button btnProgress;
    TextView textView;
    Button btnCamera;

    private static final int IMAGE_PICK_CODE = 1000;
    private static final int PERMISSION_CODE = 1001;
    private static final int IMAGE_CAMERA_CODE = 1002;
    private static final int PERMISSION_CAMERA_CODE = 1003;

    Bitmap eyePatchBitmap;
    Bitmap flowerLine;
    Canvas canvas;

    Uri mImageCaptureUri;
    Boolean processingGranted = false;

    Paint rectPaint = new Paint();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = (ImageView)findViewById(R.id.imageView);
        btnUpload = (Button) findViewById(R.id.btnUpload);
        btnProgress = (Button)findViewById(R.id.btnProgress);
        textView = (TextView) findViewById(R.id.textView);
        btnCamera=(Button)findViewById(R.id.btnCamera);

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

        btnCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
                    if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
                        // permision not granted, request it
                        String [] permissions = {Manifest.permission.CAMERA};
                        // show popup for runtime permission
                        requestPermissions(permissions,PERMISSION_CAMERA_CODE);

                    }
                    else{
                        // permission already granted
                        takePicture();

                    }
                }
                else{
                    // system os is less then marshmallow
                    takePicture();
                }

            }
        });



        btnProgress.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(processingGranted){
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

                    if(sparseArray.size() == 0 ){
                        Toast.makeText(MainActivity.this, "there is no one in this picture", Toast.LENGTH_SHORT).show();
                    }else{
                        Face face = sparseArray.valueAt(0);
                        int x=(int)face.getPosition().x;
                        int y =(int)face.getPosition().y;
                        int x2 = (int) face.getWidth();
                        int y2=(int)face.getHeight();
                        RectF rectF = new RectF(x,y,x2,y2);
                        canvas.drawRoundRect(rectF,2,2,rectPaint);
                        int x1 = x - x2*4/10;
                        int y1 = y - y2*4/10;
                        int width = x2 + x2*8/10 ;
                        int height = y2 + y2*8/10;
                        if(x1<0){ x1=0;}
                        if(y1<0){y1=0;}
                        if(x1+width > myBitmap.getWidth()){width = myBitmap.getWidth() - x1;}
                        if(y1+height > myBitmap.getHeight()){height = myBitmap.getHeight() - y1;}
                        Bitmap cropedBitmap = Bitmap.createBitmap(myBitmap,x1,y1,width,height);
                        imageView.setImageBitmap(cropedBitmap);
                        imageProcessing(cropedBitmap);
                    }
                    processingGranted = false;
                }
                else{
                    Toast.makeText(MainActivity.this, "take or upload a picture", Toast.LENGTH_SHORT).show();
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

        String textToDisplay = "";


        // GENDER DETECTION

        Interpreter tflite = null;
        try{
            tflite = new Interpreter(loadModelFile(MainActivity.this,"genderModel.tflite"));
        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);

        }

        // Create a container for the result and specify that this is a quantized model.
        // Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
        int[] probabilityShape = tflite.getOutputTensor(0).shape();
        DataType probaDataType = tflite.getOutputTensor(0).dataType();

        TensorBuffer probabilityBuffer =
                TensorBuffer.createFixedSize(probabilityShape, probaDataType);

        // Running inference
        if( tflite != null) {
            tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer().rewind());
            if(probabilityBuffer.getFloatArray()[0]>0.5)
                textToDisplay = "male";
            else
                textToDisplay = "female";
        }

        // AGE DETECTION

        Interpreter tflite2 = null;

        try{
            tflite2 = new Interpreter(loadModelFile(MainActivity.this,"age_recognition.tflite"));
        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);

        }

        probabilityShape = tflite.getOutputTensor(0).shape();
        probaDataType = tflite.getOutputTensor(0).dataType();

        TensorBuffer pb =
                TensorBuffer.createFixedSize(probabilityShape, probaDataType);

        String age = "";

        // Running inference
        if( tflite != null) {
            tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer().rewind());
            switch (getIndexOfLargest(pb.getFloatArray())) {
                case 0: age = "0-19 yo"; break;
                case 1: age = "20-39 yo"; break;
                case 2: age = "40-59 yo"; break;
                case 3: age = "60-79 yo"; break;
                case 4: age = "80-99 yo"; break;
            }
            textToDisplay += ' ' + age;
        }

        textView.setText(textToDisplay);
    }

    private void pickImageFromGallery(){
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, IMAGE_PICK_CODE);
    }

    private void takePicture(){
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent,IMAGE_CAMERA_CODE);
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
                break;
            }
            case PERMISSION_CAMERA_CODE:{
                if(grantResults.length > 0 && grantResults[0]==PackageManager.PERMISSION_GRANTED){
                    takePicture();
                }
                else{
                    //permission denied
                    Toast.makeText(this,"Permission denied!",Toast.LENGTH_SHORT).show();
                }
                break;
            }

        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode,resultCode,data);
        if(resultCode == RESULT_OK && requestCode == IMAGE_PICK_CODE){
            //set image to image view
            imageView.setImageURI(data.getData());
            processingGranted = true;
            textView.setText("");
        }
        if (requestCode == IMAGE_CAMERA_CODE){
            Bitmap bitmap = (Bitmap) data.getExtras().get("data");
            bitmap = rotateImage(bitmap,0);
            imageView.setImageBitmap(bitmap);
            processingGranted = true;
            textView.setText("");
        }
    }


    public static Bitmap rotateImage(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(),
                matrix, true);
    }

    public int getIndexOfLargest( float[] array )
    {
        if ( array == null || array.length == 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {
            if ( array[i] > array[largest] ) largest = i;
        }
        return largest; // position of the first largest found
    }
}

