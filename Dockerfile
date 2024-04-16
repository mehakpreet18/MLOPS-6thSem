from python:3.9
WORKDIR /app1
COPY dpas__model_.py /app1/
COPY knn_model.joblib /app1/
COPY Training.csv /app1/
COPY Testing.csv /app1/
RUN pip install numpy scikit-learn pandas joblib
CMD ["python", "dpas__model_.py", "knn_model.joblib", "Training.csv", "Testing.csv"]
