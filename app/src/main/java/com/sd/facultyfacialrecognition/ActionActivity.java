package com.sd.facultyfacialrecognition;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.firestore.FirebaseFirestore;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class ActionActivity extends AppCompatActivity {

    private String profName;
    private String currentLab;
    private FirebaseFirestore db;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_action);

        profName = getIntent().getStringExtra("profName");
        currentLab = getIntent().getStringExtra("currentLab");
        db = FirebaseFirestore.getInstance();

        TextView actionPromptText = findViewById(R.id.text_action_prompt);
        Button breakButton = findViewById(R.id.btn_break);
        Button endClassButton = findViewById(R.id.btn_end_class);

        if (profName != null) {
            actionPromptText.setText("Hello Professor " + profName);
        }

        breakButton.setOnClickListener(v -> onTakeBreakClicked());
        endClassButton.setOnClickListener(v -> onEndClassClicked());
    }

    private void onTakeBreakClicked() {
        if (profName == null) return;

        String facultyStatus = "Break";
        String doorStatus = "UNLOCKED"; // Door remains unlocked for a break
        String timestamp = getCurrentTimestamp();

        logDoorEvent(profName, facultyStatus, doorStatus);
        updateLabStatus(profName, facultyStatus, doorStatus, timestamp);
        updateRealtimeStatus(profName, facultyStatus, doorStatus);

        Intent intent = new Intent(ActionActivity.this, DashboardActivity.class);
        intent.putExtra("profName", profName);
        intent.putExtra("status", "Professor is on break. Please scan to resume class.");
        startActivity(intent);
        finish();
    }

    private void onEndClassClicked() {
        if (profName == null) return;

        String facultyStatus = "End Class";
        String doorStatus = "LOCKED";
        String timestamp = getCurrentTimestamp();

        logDoorEvent(profName, facultyStatus, doorStatus);
        updateLabStatus(profName, facultyStatus, doorStatus, timestamp);
        updateRealtimeStatus(profName, facultyStatus, doorStatus);

        Intent intent = new Intent(ActionActivity.this, ThankYouActivity.class);
        intent.putExtra("message", "Class ended and door is locked, thank you!");
        startActivity(intent);
        finish();
    }

    private String getCurrentTimestamp() {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd | EEEE | HH:mm:ss", Locale.getDefault());
        return sdf.format(new Date());
    }

    private void logDoorEvent(String facultyName, String facultyStatus, String doorStatus) {
        if (facultyName == null || facultyName.equals("Scanning...") || facultyName.equals("Unknown")) {
            return;
        }
        String timestamp = getCurrentTimestamp();
        Map<String, Object> logEntry = new HashMap<>();
        logEntry.put("facultyName", facultyName);
        logEntry.put("facultyStatus", facultyStatus);
        logEntry.put("doorStatus", doorStatus);
        logEntry.put("timestamp", timestamp);
        logEntry.put("lab", currentLab);

        db.collection("DoorLogs")
                .add(logEntry)
                .addOnSuccessListener(docRef -> Log.d("DoorLockDebug", "Door event logged from ActionActivity"))
                .addOnFailureListener(e -> Log.e("DoorLockDebug", "Error logging door event from ActionActivity", e));
    }

    private void updateLabStatus(String facultyName, String facultyStatus, String doorStatus, String timestamp) {
        if (facultyName == null || facultyName.equals("Scanning...") || facultyName.equals("Unknown")) return;
        Map<String, Object> data = new HashMap<>();
        data.put("facultyName", facultyName);
        data.put("facultyStatus", facultyStatus);
        data.put("doorStatus", doorStatus);
        data.put("timestamp", timestamp);

        db.collection(currentLab)
                .document("Latest")
                .set(data)
                .addOnSuccessListener(aVoid -> Log.d("DoorLockDebug", "Updated " + currentLab + " Latest from ActionActivity"))
                .addOnFailureListener(e -> Log.e("DoorLockDebug", "Error updating " + currentLab + " Latest from ActionActivity", e));
    }

    private void updateRealtimeStatus(String facultyName, String facultyStatus, String doorStatus) {
        if (profName == null || profName.equals("Scanning...") || profName.equals("Unknown")) {
            return;
        }

        String timestamp = new SimpleDateFormat("yyyy-MM-dd | EEEE | HH:mm:ss", Locale.getDefault()).format(new Date());
        Map<String, Object> data = new HashMap<>();
        data.put("facultyStatus", facultyStatus);
        data.put("facultyName", profName);
        data.put("doorStatus", doorStatus);
        data.put("timestamp", timestamp);

        try {
            FirebaseDatabase database = FirebaseDatabase.getInstance("https://facultyfacialrecognition-default-rtdb.asia-southeast1.firebasedatabase.app/");
            DatabaseReference dbRef = database.getReference(currentLab).child("Latest");
            dbRef.setValue(data)
                    .addOnSuccessListener(aVoid -> Log.d("DoorDebug", "Realtime DB successfully updated from ActionActivity"))
                    .addOnFailureListener(e -> Log.e("DoorDebug", "Realtime DB update FAILED from ActionActivity", e));
        } catch (Exception e) {
            Log.e("DoorDebug", "Database initialization error in ActionActivity", e);
        }
    }
}
