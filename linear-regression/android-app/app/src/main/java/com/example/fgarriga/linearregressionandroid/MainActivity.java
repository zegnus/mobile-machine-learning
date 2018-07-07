package com.example.fgarriga.linearregressionandroid;

import android.app.Activity;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends Activity {

    private static final String MODEL_NAME = "file:///android_asset/optimized_frozen_linear_regression.pb";
    private static final String INPUT_NODE = "x";
    private static final String OUTPUT_NODE = "y_output";
    private static final long[] INPUT_SHAPE = {1L, 1L};

    private static TensorFlowInferenceInterface tensorFlowInferenceInterface;

    private EditText editText;
    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        editText = findViewById(R.id.edit_text);
        textView = findViewById(R.id.text_view);

        tensorFlowInferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_NAME);
    }

    public void pressButton(View view) {
        float input = Float.parseFloat(editText.getText().toString());
        String results = performInference(input);
        textView.setText(results);
    }

    private String performInference(float input) {
        float[] floatArray = {input};

        tensorFlowInferenceInterface.feed(INPUT_NODE, floatArray, INPUT_SHAPE);
        tensorFlowInferenceInterface.run(new String[] {OUTPUT_NODE});

        float[] results = {0.0f};
        tensorFlowInferenceInterface.fetch(OUTPUT_NODE, results);

        return String.valueOf(results[0]);
    }
}
