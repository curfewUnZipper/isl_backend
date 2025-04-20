# this is generating onnx model, and is giving classname:numebrs (probably percentsage liek 0.012233232)

import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import tensorflow as tf
def saveAsTflite(model,X):    
    # Step 1: Convert scikit-learn model to ONNX
    X = np.array(X)  # Assuming X is your input data (e.g., a feature matrix)
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]  # Define input type
    onnx_model = convert_sklearn(model, initial_types=initial_type)  # Convert scikit-learn pipeline to ONNX

    # Save the ONNX model
    onnx.save_model(onnx_model, "model.onnx")




    # Step 2: Load and run inference using ONNX Runtime
    # Initialize the ONNX Runtime session
    session = ort.InferenceSession("model.onnx")

    # Prepare input data (replace input_data with actual input data for prediction)
    input_data = np.array(X, dtype=np.float32)  # Example input for inference
    input_name = session.get_inputs()[0].name

    # Run inference
    result = session.run(None, {input_name: input_data})

    # Print the inference result
    print("Inference Result:", result)

    # Step 3 (optional): Convert ONNX to TensorFlow (using tf2onnx)
    # Ensure that the ONNX model is valid
    onnx.checker.check_model(onnx_model)

    # Use tf2onnx for converting ONNX to TensorFlow
    import tf2onnx
    tf_rep = tf2onnx.convert.from_onnx(onnx_model)

    # Step 4: Save the TensorFlow model
    tf_rep.export_graph("model_tf")

    # Step 5: Convert TensorFlow model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")  # Load the model
    tflite_model = converter.convert()  # Convert to TFLite

    # Save the TFLite model
    with open("./artifacts/model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Saved model as model.tflite")


