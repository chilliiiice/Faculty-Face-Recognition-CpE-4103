package com.sd.facultyfacialrecognition;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class ThankYouActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_thank_you);

        String message = getIntent().getStringExtra("message");
        TextView msg = findViewById(R.id.text_goodbye);
        msg.setText(message);

        new Handler().postDelayed(() -> {
            // Return to face recognition screen
            Intent intent = new Intent(ThankYouActivity.this, MainActivity.class);
            startActivity(intent);
            finish();
        }, 3000);
    }
}
