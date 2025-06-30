#!$SHELL_CMDù
# filepath: setup_docker.sh

echo "🚀 SCVSAI - Setup Docker Environment"
echo "=================================="

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
IMAGE_NAME="scvsai:latest"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OS" == "Windows_NT" ]]; then
    SHELL_CMD="bash"
    print_info "Windows environment detected - using bash"
else
    SHELL_CMD="/bin/bash"
    print_info "Unix environment detected - using /bin/bash"
fi

# Funzione per stampare messaggi colorati
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Funzione per controllare se un comando esiste
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Controlla se Docker è installato
print_info "Checking Docker installation..."
if command_exists docker; then
    print_success "Docker is already installed"
    docker --version
else
    print_warning "Docker not found. Installing Docker..."
    
    # Detecta OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_info "Installing Docker on Linux..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        print_warning "Please logout and login again to use Docker without sudo"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_warning "Please install Docker Desktop for Mac from: https://www.docker.com/products/docker-desktop"
        exit 1
    else
        print_error "Unsupported operating system"
        exit 1
    fi
fi

# 2. Verifica che Docker sia in esecuzione
print_info "Checking if Docker daemon is running..."
if docker info >/dev/null 2>&1; then
    print_success "Docker daemon is running"
else
    print_error "Docker daemon is not running. Please start Docker and run this script again."
    exit 1
fi

print_info "Checking if Docker image exists..."
if docker images | grep -q "scvsai.*latest"; then
    print_success "Docker image 'scvsai:latest' already exists"
    
    # Chiedi se rebuildarla
    echo ""
    read -p "Do you want to rebuild the image? (y/N): " rebuild_choice
    case $rebuild_choice in
        [Yy]* )
            print_info "Rebuilding Docker image..."
            if docker build -t $IMAGE_NAME .; then
                print_success "Docker image rebuilt successfully: $IMAGE_NAME"
            else
                print_error "Failed to rebuild Docker image"
                exit 1
            fi
            ;;
        * )
            print_info "Using existing Docker image"
            ;;
    esac
else
    # Build dell'immagine se non esiste
    print_info "Building Docker image for the first time..."
    if docker build -t $IMAGE_NAME .; then
        print_success "Docker image built successfully: $IMAGE_NAME"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
fi

# 4. Verifica che l'immagine sia disponibile
print_info "Verifying Docker image availability..."
if docker images | grep -q "scvsai.*latest"; then
    print_success "Docker image verification passed"
else
    print_error "Docker image not found after build/check"
    exit 1
fi

# 5. Controlla se ci sono container in esecuzione
print_info "Checking for running containers..."
RUNNING_CONTAINERS=$(docker ps --filter "ancestor=$IMAGE_NAME" --format "table {{.ID}}\t{{.Status}}" | tail -n +2)

if [ ! -z "$RUNNING_CONTAINERS" ]; then
    print_warning "Found running containers:"
    echo "$RUNNING_CONTAINERS"
    echo ""
    read -p "Do you want to stop running containers? (y/N): " stop_choice
    case $stop_choice in
        [Yy]* )
            print_info "Stopping running containers..."
            docker ps --filter "ancestor=$IMAGE_NAME" -q | xargs -r docker stop
            print_success "Containers stopped"
            ;;
        * )
            print_info "Keeping containers running"
            ;;
    esac
fi

# 6. Menu interattivo
echo ""
echo "🎉 Setup completed successfully!"
echo ""
print_info "Available commands:"
echo "  1) Enter Docker shell (interactive)"
echo "  2) Run EDA analysis"
echo "  3) Run training"
echo "  4) Run tests"
echo "  5) Run complete workflow"
echo "  6) Show Docker commands help"
echo "  7) Force rebuild image"
echo "  8) Clean up (remove image and containers)"
echo "  9) Exit"
echo ""

while true; do
    read -p "Select an option (1-9): " choice
    case $choice in
        1)
            print_info "Entering Docker shell..."
            echo "You can now use 'make' commands inside the container:"
            echo "  make eda      - Run EDA"
            echo "  make train    - Run training" 
            echo "  make test     - Run tests"
            echo "  make workflow - Run all"
            echo "  exit          - Exit container"
            echo ""
            docker run --rm -it \
                -v "$(pwd)/models:/app/models" \
                -v "$(pwd)/data:/app/data" \
                "$IMAGE_NAME" $SHELL_CMD
            ;;
        2)
            print_info "Running EDA analysis..."
            docker run --rm \
                -v "$(pwd)/data:/app/data" \
                $IMAGE_NAME make eda
            print_success "EDA completed. Check the data/ folder for results."
            ;;
        3)
            print_info "Running training..."
            docker run --rm \
                -v "$(pwd)/models:/app/models" \
                -v "$(pwd)/data:/app/data" \
                $IMAGE_NAME make train
            print_success "Training completed. Check the models/ folder for results."
            ;;
        4)
            print_info "Running tests..."
            docker run --rm $IMAGE_NAME make test
            ;;
        5)
            print_info "Running complete workflow..."
            docker run --rm \
                -v "$(pwd)/models:/app/models" \
                -v "$(pwd)/data:/app/data" \
                $IMAGE_NAME make workflow
            print_success "Complete workflow finished!"
            ;;
        6)
            echo ""
            print_info "Docker Commands Reference:"
            echo "# Build image:"
            echo "docker build -t scvsai:latest ."
            echo ""
            echo "# Run interactive shell:"
            echo "docker run --rm -it -v \$(pwd)/models:/app/models -v \$(pwd)/data:/app/data scvsai:latest $SHELL_CMD"
            echo ""
            echo "# Run specific commands:"
            echo "docker run --rm -v \$(pwd)/data:/app/data scvsai:latest make eda"
            echo "docker run --rm -v \$(pwd)/models:/app/models scvsai:latest make train"
            echo "docker run --rm scvsai:latest make test"
            echo ""
            ;;
        7)
            print_info "Force rebuilding Docker image..."
            docker build --no-cache -t $IMAGE_NAME .
            print_success "Docker image force rebuilt successfully!"
            ;;
        8)
            print_warning "This will remove the Docker image and stop all containers!"
            read -p "Are you sure? (y/N): " cleanup_choice
            case $cleanup_choice in
                [Yy]* )
                    print_info "Cleaning up..."
                    # Stop containers
                    docker ps --filter "ancestor=$IMAGE_NAME" -q | xargs -r docker stop
                    # Remove containers  
                    docker ps -a --filter "ancestor=$IMAGE_NAME" -q | xargs -r docker rm
                    # Remove image
                    docker rmi $IMAGE_NAME 2>/dev/null || true
                    print_success "Cleanup completed!"
                    ;;
                * )
                    print_info "Cleanup cancelled"
                    ;;
            esac
            ;;
        9)
            print_success "Goodbye! 👋"
            exit 0
            ;;
        *)
            print_warning "Invalid option. Please select 1-9."
            ;;
    esac
    echo ""
done