# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files
# and to ensure output is sent straight to the terminal (unbuffered)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# Copying this separately leverages Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Set the default command to run your CLI application.
# We use `python -u` to force unbuffered output, which is essential
# for interactive CLI apps in Docker[reference:1].
CMD ["python", "-u", "cli.py"]