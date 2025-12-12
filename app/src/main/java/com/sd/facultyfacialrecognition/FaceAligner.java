package com.sd.facultyfacialrecognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.util.List;

/**
 * FaceAligner detects a face in a bitmap, crops it, and resizes it to 160x160 for FaceNet.
 */
public class FaceAligner {

    private final FaceDetector detector;

    public FaceAligner(@NonNull Context context) {
        // High-speed face detection
        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                .build();

        detector = FaceDetection.getClient(options);
    }

    /**
     * Detects the first face in the bitmap, crops it, and resizes to 160x160.
     * Returns null if no face is found.
     */
    public Bitmap alignFace(Bitmap bitmap) {
        try {
            InputImage image = InputImage.fromBitmap(bitmap, 0);

            // Blocking call for simplicity
            List<Face> faces = Tasks.await(detector.process(image));

            if (faces.size() == 0) {
                Log.d("FaceAligner", "No face detected.");
                return null;
            }

            // Take the first detected face
            Rect bounds = faces.get(0).getBoundingBox();

            // Ensure bounds are within bitmap dimensions
            int left = Math.max(bounds.left, 0);
            int top = Math.max(bounds.top, 0);
            int width = Math.min(bounds.width(), bitmap.getWidth() - left);
            int height = Math.min(bounds.height(), bitmap.getHeight() - top);

            Bitmap faceCrop = Bitmap.createBitmap(bitmap, left, top, width, height);

            // Resize to 160x160 for FaceNet
            return Bitmap.createScaledBitmap(faceCrop, 160, 160, true);

        } catch (Exception e) {
            e.printStackTrace();
            Log.e("FaceAligner", "Face alignment failed: " + e.getMessage());
            return null;
        }
    }

    /**
     * Close the detector when done.
     */
    public void close() {
        try {
            detector.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
