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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
import tensorflow as tf
from biopandas.pdb import PandasPdb
from tensorflow import keras
from tensorflow.keras import layers


# def AFD(self, teacher_features, student_features):
#     # Calculate queries and keys using global average pooling
#     query_teacher = tf.reduce_mean(teacher_features, axis=[1, 2])  # Assuming height and width axes are 1 and 2
#     key_student = tf.reduce_mean(student_features, axis=[1, 2])

#     # Calculate attention scores
#     attention_scores = calculate_attention(query_teacher, key_student)

#     # Compute distillation loss based on attention scores
#     distillation_loss = self.distillation_loss_fn(
#         teacher_features, student_features, attention_scores)

#     return distillation_loss


# def attention_based_feature_distillation_loss(teacher_features, student_features):
#     # Normalize features
#     teacher_features = tf.nn.l2_normalize(teacher_features, axis=-1)
#     student_features = tf.nn.l2_normalize(student_features, axis=-1)

#     # Calculate cosine similarity between teacher and student features
#     similarity = tf.keras.losses.CosineSimilarity(axis=-1)
#     similarity_scores = similarity(teacher_features, student_features)

#     # Compute the mean of similarity scores as the distillation loss
#     distillation_loss = 1 - tf.reduce_mean(similarity_scores)

#     return distillation_loss


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
        beta=200,
        temperature=10,
    ):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.beta = beta
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
            
            # Calculate attention-based feature distillation loss
            afd_loss = self.attention_based_feature_distillation(teacher_predictions, student_predictions)
            loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss + self.beta * afd_loss
            
            #  # Calculate attention-based feature distillation loss
            # afd_loss = self.attention_based_feature_distillation(x, teacher_predictions, student_predictions)
            
            # Compute the total loss
            # loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss

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
    
    
    def attention_based_feature_distillation(self, teacher_features, student_features):
        # Tính toán sự tương đồng giữa các đặc trưng của giáo viên và học sinh
        similarity_matrix = self.calculate_similarity(teacher_features, student_features)

        # Tính toán loss dựa trên similarity matrix
        afd_loss = self.compute_afd_loss(similarity_matrix)
        
        return afd_loss

    def calculate_similarity(self, teacher_features, student_features):
        # Cosine similarity
        teacher_normalized = tf.nn.l2_normalize(teacher_features, axis=1)
        student_normalized = tf.nn.l2_normalize(student_features, axis=1)
        similarity_matrix = tf.matmul(teacher_normalized, student_normalized, transpose_b=True)
        
        return similarity_matrix

    def compute_afd_loss(self, similarity_matrix):
        # Mean squared error giữa similarity matrix
        afd_loss = tf.reduce_mean(tf.square(similarity_matrix))
        
        return afd_loss
    
