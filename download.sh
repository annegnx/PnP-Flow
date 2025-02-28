FILE=$1

if [ $FILE == "pretrained-network-celeba" ]; then
    OUTPUT_DIR=./model/celeba/ot
    mkdir -p $OUTPUT_DIR
    gdown --id 1ZZ6S-PGRx-tOPkr4Gt3A6RN-PChabnD6 -O $OUTPUT_DIR/model_final.pt

elif  [ $FILE == "pretrained-network-afhq-cat" ]; then
    OUTPUT_DIR=./model/afhq_cat/ot
    mkdir -p $OUTPUT_DIR
    gdown --id 1FpD3cYpgtM8-KJ3Qk48fcjtr1Ne_IMOF -O $OUTPUT_DIR/model_final.pt

elif  [ $FILE == "celeba-hq-dataset" ]; then
    DEST_DIR=./data/celeba_hq
    mkdir -p $DEST_DIR
    URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
    ZIP_FILE=./data/celeba_hq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d $DEST_DIR
    rm $ZIP_FILE

elif  [ $FILE == "afhq-cat-dataset" ]; then
    DEST_DIR=./data
    mkdir -p $DEST_DIR
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    ZIP_FILE=./data/afhq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d $DEST_DIR
    rm $ZIP_FILE
    mv ./data/afhq ./data/afhq_cat
    # bash scripts/afhq_validation_images.sh

elif  [ $FILE == "celeba-dataset" ]; then
    DEST_DIR=./data/celeba
    ZIP_FILE="$DEST_DIR/celeba-dataset.zip"
    mkdir -p $DEST_DIR
    echo "Downloading CelebA dataset..."
    kaggle datasets download jessicali9530/celeba-dataset -p "$DEST_DIR"
    # Ensure the ZIP file exists before extracting
    if [ -f "$ZIP_FILE" ]; then
        echo "Dataset downloaded. Extracting..."
        unzip -q "$ZIP_FILE" -d "$DEST_DIR"
        rm "$ZIP_FILE"
        echo "Extraction completed!"
    else
        echo "Error: ZIP file not found after download!"
        exit 1
    mv ./data/celeba/img_align_celeba/img_align_celeba/* ./data/celeba
    rm -r ./data/celeba/img_align_celeba
    fi
else
    echo "Available arguments are pretrained-network-celeba, pretrained-network-afhq-cat, celeba-dataset, celeba-hq-dataset, and afhq-cat-dataset."
    exit 1
fi