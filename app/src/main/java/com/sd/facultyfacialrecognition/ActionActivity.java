package com.sd.facultyfacialrecognition;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class ActionActivity extends AppCompatActivity {

    private TextView actionPrompt;
    private Button breakButton, endClassButton;
    private String profName;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_action);

        profName = getIntent().getStringExtra("profName");

        actionPrompt = findViewById(R.id.text_action_prompt);
        breakButton = findViewById(R.id.btn_break);
        endClassButton = findViewById(R.id.btn_end_class);

        actionPrompt.setText("Hello Prof. " + profName + ", choose an action:");

        breakButton.setOnClickListener(v -> {
            Intent intent = new Intent(ActionActivity.this, DashboardActivity.class);
            intent.putExtra("profName", profName);
            intent.putExtra("status", "Prof. " + profName + " is on break.");
            startActivity(intent);
        });

        endClassButton.setOnClickListener(v -> {
            Intent intent = new Intent(ActionActivity.this, ThankYouActivity.class);
            intent.putExtra("message", "Class ended, thank you.");
            startActivity(intent);
        });

    }
}
