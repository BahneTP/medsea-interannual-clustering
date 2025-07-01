# MedSea Spatio-Temporal Clustering

This repository contains code and experiments for clustering spatio-temporal oceanographic data in the Mediterranean Sea.

## Overview

Each timestamp represents a full spatial snapshot of the Mediterranean, including salinity and temperature at three depths: 50 m, 300 m, and 1000 m. The goal is to learn compact latent representations of these maps and group them into clusters that capture the main interannual patterns and changes.

## Models

We train and compare four different autoencoder-based architectures:
- **AE** (Autoencoder)
- **VAE** (Variational Autoencoder)
- **CAE** (Convolutional Autoencoder)
- **VCAE** (Variational Convolutional Autoencoder)

These models compress the spatio-temporal input data into a latent space, which is then clustered into nine clusters to analyze how the Mediterranean evolves over time.

Additionally, standard **KMeans** clustering is applied for baseline comparison.

## Main Features

- Handles missing or irrelevant regions with masking.
- Combines spatial feature extraction (convolutions) and latent regularization (variational) for robust clustering.
- Visualizes the temporal evolution of clusters.

## Data

Input:  
- Gridded salinity and temperature data for all locations in the Mediterranean.
- Multiple time snapshots spanning several years.

## Acknowledgements

This work was carried out in collaboration with the **University of Split** as part of a research internship, with support from  
Frano Matić (Department of Marine Studies) and 
Hrvoje Kalinić (Department of Informatics).

## License

MIT — use freely, but cite if useful.
