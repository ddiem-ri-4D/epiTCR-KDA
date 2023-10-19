import sklearn.metrics as metrics

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,roc_auc_score,classification_report,roc_curve, auc
from sklearn.metrics import classification_report
from keras.models import load_model

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import functools

import numpy as np
import pandas as pd
import random
import glob
import itertools
from biopandas.pdb import PandasPdb
from tensorflow import keras
from tensorflow.keras import layers


# Create the Distiller class
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=10,
    ):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack the data
        x, y = data

        # Forward pass through the teacher model
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass through the student model
            student_predictions = self.student(x, training=True)

            # Compute the student loss
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute the distillation loss
            distillation_loss = self.distillation_loss_fn(
                keras.activations.softmax(teacher_predictions / self.temperature, axis=1),
                keras.activations.softmax(student_predictions / self.temperature, axis=1),
            )

            # Compute the total loss
            loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss

        # Compute the gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update the student model's weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dictionary of metrics
        return {m.name: m.result() for m in self.metrics}
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Convert labels to binary format
        y_binary = tf.cast(y < 5, dtype=tf.float32)  # Example threshold for binary classification

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y_binary, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y_binary, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
    
