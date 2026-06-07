from src.cnn.predict import predict_xray

def prediction_agent(state):

    predictions = predict_xray(
        state["image_path"]
    )

    state["predictions"] = predictions

    return state