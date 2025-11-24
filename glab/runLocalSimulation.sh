#!/bin/bash
SIMULATION_TYPE="local-simulation-gpu"
FLWR_TELEMETRY=0
RAY_DEDUPLICATION_LOGS=0

# Run the Flower simulation with the specified parameters and environment variables
FLWR_TELEMETRY_ENABLED=${FLWR_TELEMETRY} RAY_DEDUP_LOGS=${RAY_DEDUPLICATION_LOGS} flwr run . ${SIMULATION_TYPE} --stream