import onnxruntime as ort


def load_model(model_path):
    session = ort.InferenceSession(model_path)
    return session