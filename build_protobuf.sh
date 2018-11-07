THIS_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

protoc --proto_path="${THIS_FILE_DIR}/data/shapenet" --python_out="${THIS_FILE_DIR}/data/shapenet" "${THIS_FILE_DIR}/data/shapenet/meta.proto"