echo "Getting Training Images for the fashion MNIST..."
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz" --no-check-certificate
echo "Training Data Successfully fetched!"

echo "Getting Training Labels for the fashion MNIST..."
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz" --no-check-certificate
echo "Training Labels Successfully fetched!"

echo "Getting Test Images for the fashion MNIST..."
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz" --no-check-certificate
echo "Training Data Successfully fetched!"

echo "Getting Test Labels for the fashion MNIST..."
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz" --no-check-certificate
echo "Training Data Successfully fetched!"

echo "Fashion MNIST Dataset Successfully fetched!!"

echo "Starting to unzip the fetched files..."
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
echo "Fetched files unzipped successfully!!"
