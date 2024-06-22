import cv2
from sklearn.cluster import KMeans
import numpy as np


class TeamAssigner:

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}  #dictionary where we will have a player id and weather they are in team 1 or 2

    def get_clustering_model(self, image):
        #Reshape the image into 2D
        image_2d = image.reshape(-1,3)
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])]
        image_top_half = image[0:image[(image.shape)//2]]

        kmeans = self.get_clustering_model(image_top_half)

        labels = kmeans.labels_
        clustered_image = labels.reshape(image.shape[0], image.shape[1])

        #finding the background and player labels
        corner_clusters = [clustered_image[0,0,], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1, -1]]
        background = max(set(corner_clusters), key = corner_clusters.count)
        player = 1 - background

        player_color = kmeans.cluster_centers_[player]
        
        return player_color



    def assign_team_color(self, frame, player_detections):

        player_colors = []
        for _, player_detections in player_detections.items():
            bbox = player_detections["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init = 'k-means++', n_init = 1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        ##################