# Use an official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt before copying the whole project
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
