#!/bin/bash
set -euo pipefail

# ============================================================================
# Zen MCP Server Setup Script
# 
# A platform-agnostic setup script that works on macOS, Linux, and WSL.
# Handles environment setup, dependency installation, and configuration.
# ============================================================================

# ----------------------------------------------------------------------------
# Constants and Configuration
# ----------------------------------------------------------------------------

# Colors for output (ANSI codes work on all platforms)
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly RED='\033[0;31m'
readonly NC='\033[0m' # No Color

# Configuration
readonly VENV_PATH=".zen_venv"
readonly DOCKER_CLEANED_FLAG=".docker_cleaned"
readonly LOG_DIR="logs"
readonly LOG_FILE="mcp_server.log"

# ----------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------

# Print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

# Get the script's directory (works on all platforms)
get_script_dir() {
    cd "$(dirname "$0")" && pwd
}

# Extract version from config.py
get_version() {
    grep -E '^__version__ = ' config.py 2>/dev/null | sed 's/__version__ = "\(.*\)"/\1/' || echo "unknown"
}

# ----------------------------------------------------------------------------
# Platform Detection Functions
# ----------------------------------------------------------------------------

# Detect the operating system
detect_os() {
    case "$OSTYPE" in
        darwin*)  echo "macos" ;;
        linux*)   
            if grep -qi microsoft /proc/version 2>/dev/null; then
                echo "wsl"
            else
                echo "linux"
            fi
            ;;
        msys*|cygwin*|win32) echo "windows" ;;
        *)        echo "unknown" ;;
    esac
}

# Get Claude config path based on platform
get_claude_config_path() {
    local os_type=$(detect_os)
    
    case "$os_type" in
        macos)
            echo "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
            ;;
        linux)
            echo "$HOME/.config/Claude/claude_desktop_config.json"
            ;;
        wsl)
            echo "/mnt/c/Users/$USER/AppData/Roaming/Claude/claude_desktop_config.json"
            ;;
        windows)
            echo "$APPDATA/Claude/claude_desktop_config.json"
            ;;
        *)
            echo ""
            ;;
    esac
}

# ----------------------------------------------------------------------------
# Docker Cleanup Functions
# ----------------------------------------------------------------------------

# Clean up old Docker artifacts
cleanup_docker() {
    # Skip if already cleaned or Docker not available
    [[ -f "$DOCKER_CLEANED_FLAG" ]] && return 0
    
    if ! command -v docker &> /dev/null || ! docker info &> /dev/null 2>&1; then
        return 0
    fi
    
    local found_artifacts=false
    
    # Define containers to remove
    local containers=(
        "gemini-mcp-server"
        "gemini-mcp-redis"
        "zen-mcp-server"
        "zen-mcp-redis"
        "zen-mcp-log-monitor"
    )
    
    # Remove containers
    for container in "${containers[@]}"; do
        if docker ps -a --format "{{.Names}}" | grep -q "^${container}$" 2>/dev/null; then
            if [[ "$found_artifacts" == false ]]; then
                echo "One-time Docker cleanup..."
                found_artifacts=true
            fi
            echo "  Removing container: $container"
            docker stop "$container" >/dev/null 2>&1 || true
            docker rm "$container" >/dev/null 2>&1 || true
        fi
    done
    
    # Remove images
    local images=("gemini-mcp-server:latest" "zen-mcp-server:latest")
    for image in "${images[@]}"; do
        if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${image}$" 2>/dev/null; then
            if [[ "$found_artifacts" == false ]]; then
                echo "One-time Docker cleanup..."
                found_artifacts=true
            fi
            echo "  Removing image: $image"
            docker rmi "$image" >/dev/null 2>&1 || true
        fi
    done
    
    # Remove volumes
    local volumes=("redis_data" "mcp_logs")
    for volume in "${volumes[@]}"; do
        if docker volume ls --format "{{.Name}}" | grep -q "^${volume}$" 2>/dev/null; then
            if [[ "$found_artifacts" == false ]]; then
                echo "One-time Docker cleanup..."
                found_artifacts=true
            fi
            echo "  Removing volume: $volume"
            docker volume rm "$volume" >/dev/null 2>&1 || true
        fi
    done
    
    if [[ "$found_artifacts" == true ]]; then
        print_success "Docker cleanup complete"
    fi
    
    touch "$DOCKER_CLEANED_FLAG"
}

# ----------------------------------------------------------------------------
# Python Environment Functions
# ----------------------------------------------------------------------------

# Find suitable Python command
find_python() {
    # Prefer Python 3.12 for best compatibility
    local python_cmds=("python3.12" "python3.13" "python3.11" "python3.10" "python3" "python" "py")
    
    for cmd in "${python_cmds[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            local version=$($cmd --version 2>&1)
            if [[ $version =~ Python\ 3\.([0-9]+)\.([0-9]+) ]]; then
                local major_version=${BASH_REMATCH[1]}
                local minor_version=${BASH_REMATCH[2]}
                
                # Check minimum version (3.10) for better library compatibility
                if [[ $major_version -ge 10 ]]; then
                    echo "$cmd"
                    print_success "Found Python: $version"
                    
                    # Recommend Python 3.12
                    if [[ $major_version -ne 12 ]]; then
                        print_info "Note: Python 3.12 is recommended for best compatibility."
                    fi
                    
                    return 0
                fi
            fi
        fi
    done
    
    print_error "Python 3.10+ not found. Please install Python 3.10 or newer (3.12 recommended)."
    return 1
}

# Setup virtual environment
setup_venv() {
    local python_cmd="$1"
    local venv_python=""
    
    # Create venv if it doesn't exist
    if [[ ! -d "$VENV_PATH" ]]; then
        print_info "Creating isolated environment..."
        if $python_cmd -m venv "$VENV_PATH" 2>/dev/null; then
            print_success "Created isolated environment"
        else
            print_error "Failed to create virtual environment"
            exit 1
        fi
    fi
    
    # Get venv Python path based on platform
    local os_type=$(detect_os)
    case "$os_type" in
        windows)
            venv_python="$VENV_PATH/Scripts/python.exe"
            ;;
        *)
            venv_python="$VENV_PATH/bin/python"
            ;;
    esac
    
    # Always use venv Python
    if [[ -f "$venv_python" ]]; then
        echo "$venv_python"
        if [[ -n "${VIRTUAL_ENV:-}" ]]; then
            print_success "Using activated virtual environment"
        else
            print_info "Using virtual environment directly (no activation needed)"
        fi
        return 0
    else
        print_error "Virtual environment Python not found"
        exit 1
    fi
}

# Check if package is installed
check_package() {
    local python_cmd="$1"
    local package="$2"
    $python_cmd -c "import $package" 2>/dev/null
}

# Install dependencies
install_dependencies() {
    local python_cmd="$1"
    local deps_needed=false
    
    # Check required packages
    local packages=("mcp" "google.generativeai" "openai" "pydantic")
    for package in "${packages[@]}"; do
        local import_name=${package%%.*}  # Get first part before dot
        if ! check_package "$python_cmd" "$import_name"; then
            deps_needed=true
            break
        fi
    done
    
    if [[ "$deps_needed" == false ]]; then
        print_success "Dependencies already installed"
        return 0
    fi
    
    echo ""
    print_info "Setting up Zen MCP Server..."
    echo "Installing required components:"
    echo "  • MCP protocol library"
    echo "  • AI model connectors"
    echo "  • Data validation tools"
    echo ""
    
    # Determine if we're in a venv
    local install_cmd
    if [[ -n "${VIRTUAL_ENV:-}" ]] || [[ "$python_cmd" == *"$VENV_PATH"* ]]; then
        install_cmd="$python_cmd -m pip install -q -r requirements.txt"
    else
        install_cmd="$python_cmd -m pip install -q --user -r requirements.txt"
    fi
    
    # Install packages
    echo -n "Downloading packages..."
    if $install_cmd 2>&1 | grep -i error | grep -v warning; then
        echo -e "\r${RED}✗ Setup failed${NC}                      "
        echo ""
        echo "Try running manually:"
        echo "  $python_cmd -m pip install mcp google-genai openai pydantic"
        return 1
    else
        echo -e "\r${GREEN}✓ Setup complete!${NC}                    "
        return 0
    fi
}

# ----------------------------------------------------------------------------
# Environment Configuration Functions
# ----------------------------------------------------------------------------

# Setup .env file
setup_env_file() {
    if [[ -f .env ]]; then
        print_success ".env file already exists"
        migrate_env_file
        return 0
    fi
    
    if [[ ! -f .env.example ]]; then
        print_error ".env.example not found!"
        return 1
    fi
    
    cp .env.example .env
    print_success "Created .env from .env.example"
    
    # Detect sed version for cross-platform compatibility
    local sed_cmd
    if sed --version >/dev/null 2>&1; then
        sed_cmd="sed -i"  # GNU sed (Linux)
    else
        sed_cmd="sed -i ''"  # BSD sed (macOS)
    fi
    
    # Update API keys from environment if present
    local api_keys=(
        "GEMINI_API_KEY:your_gemini_api_key_here"
        "OPENAI_API_KEY:your_openai_api_key_here"
        "XAI_API_KEY:your_xai_api_key_here"
        "OPENROUTER_API_KEY:your_openrouter_api_key_here"
    )
    
    for key_pair in "${api_keys[@]}"; do
        local key_name="${key_pair%%:*}"
        local placeholder="${key_pair##*:}"
        local key_value="${!key_name:-}"
        
        if [[ -n "$key_value" ]]; then
            $sed_cmd "s/$placeholder/$key_value/" .env
            print_success "Updated .env with $key_name from environment"
        fi
    done
    
    return 0
}

# Migrate .env file from Docker to standalone format
migrate_env_file() {
    # Check if migration is needed
    if ! grep -q "host\.docker\.internal" .env 2>/dev/null; then
        return 0
    fi
    
    print_warning "Migrating .env from Docker to standalone format..."
    
    # Create backup
    cp .env .env.backup_$(date +%Y%m%d_%H%M%S)
    
    # Detect sed version for cross-platform compatibility
    local sed_cmd
    if sed --version >/dev/null 2>&1; then
        sed_cmd="sed -i"  # GNU sed (Linux)
    else
        sed_cmd="sed -i ''"  # BSD sed (macOS)
    fi
    
    # Replace host.docker.internal with localhost
    $sed_cmd 's/host\.docker\.internal/localhost/g' .env
    
    print_success "Migrated Docker URLs to localhost in .env"
    echo "  (Backup saved as .env.backup_*)"
}

# Validate API keys
validate_api_keys() {
    local has_key=false
    local api_keys=(
        "GEMINI_API_KEY:your_gemini_api_key_here"
        "OPENAI_API_KEY:your_openai_api_key_here"
        "XAI_API_KEY:your_xai_api_key_here"
        "OPENROUTER_API_KEY:your_openrouter_api_key_here"
    )
    
    for key_pair in "${api_keys[@]}"; do
        local key_name="${key_pair%%:*}"
        local placeholder="${key_pair##*:}"
        local key_value="${!key_name:-}"
        
        if [[ -n "$key_value" ]] && [[ "$key_value" != "$placeholder" ]]; then
            print_success "$key_name configured"
            has_key=true
        fi
    done
    
    # Check custom API URL
    if [[ -n "${CUSTOM_API_URL:-}" ]]; then
        print_success "CUSTOM_API_URL configured: $CUSTOM_API_URL"
        has_key=true
    fi
    
    if [[ "$has_key" == false ]]; then
        print_error "No API keys found in .env!"
        echo ""
        echo "Please edit .env and add at least one API key:"
        echo "  GEMINI_API_KEY=your-actual-key"
        echo "  OPENAI_API_KEY=your-actual-key"
        echo "  XAI_API_KEY=your-actual-key"
        echo "  OPENROUTER_API_KEY=your-actual-key"
        echo ""
        return 1
    fi
    
    return 0
}

# ----------------------------------------------------------------------------
# Claude Integration Functions
# ----------------------------------------------------------------------------

# Check if MCP is added to Claude and verify it's correct
check_claude_integration() {
    local python_cmd="$1"
    local server_path="$2"
    
    if ! command -v claude &> /dev/null; then
        return 2  # Claude CLI not installed
    fi
    
    # Check if zen is registered
    local mcp_list=$(claude mcp list 2>/dev/null)
    if echo "$mcp_list" | grep -q "zen"; then
        # Check if it's using the old Docker command
        if echo "$mcp_list" | grep -E "zen.*docker|zen.*compose" &>/dev/null; then
            print_warning "Found old Docker-based MCP registration, updating..."
            claude mcp remove zen 2>/dev/null || true
            
            # Re-add with correct Python command
            if claude mcp add zen -s user -- "$python_cmd" "$server_path" 2>/dev/null; then
                print_success "Updated MCP 'zen' to use standalone Python"
                return 0
            else
                echo ""
                echo "Failed to update MCP registration. Please run manually:"
                echo "  claude mcp remove zen"
                echo "  claude mcp add zen -s user -- $python_cmd $server_path"
                return 1
            fi
        else
            # Verify the registered path matches current setup
            local expected_cmd="$python_cmd $server_path"
            if echo "$mcp_list" | grep -F "$server_path" &>/dev/null; then
                print_success "MCP 'zen' correctly configured"
                return 0
            else
                print_warning "MCP 'zen' registered with different path, updating..."
                claude mcp remove zen 2>/dev/null || true
                
                if claude mcp add zen -s user -- "$python_cmd" "$server_path" 2>/dev/null; then
                    print_success "Updated MCP 'zen' with current path"
                    return 0
                else
                    echo ""
                    echo "Failed to update MCP registration. Please run manually:"
                    echo "  claude mcp remove zen"
                    echo "  claude mcp add zen -s user -- $python_cmd $server_path"
                    return 1
                fi
            fi
        fi
    else
        # Not registered at all, try to add it
        echo ""
        print_info "Registering MCP 'zen' with Claude..."
        if claude mcp add zen -s user -- "$python_cmd" "$server_path" 2>/dev/null; then
            print_success "Successfully added MCP 'zen' to Claude"
            return 0
        else
            echo ""
            echo "To add to Claude CLI manually, run:"
            echo "  claude mcp add zen -s user -- $python_cmd $server_path"
            return 1
        fi
    fi
}

# Display setup instructions
display_setup_instructions() {
    local python_cmd="$1"
    local server_path="$2"
    
    echo ""
    echo "===== SETUP COMPLETE ====="
    echo ""
    echo "To use Zen MCP Server:"
    echo ""
    echo "1. For Claude Code CLI:"
    echo "   claude mcp add zen -s user -- $python_cmd $server_path"
    echo ""
    echo "2. For Claude Desktop, add to config:"
    cat << EOF
   {
     "mcpServers": {
       "zen": {
         "command": "$python_cmd",
         "args": ["$server_path"]
       }
     }
   }
EOF
    
    # Show platform-specific config location
    local config_path=$(get_claude_config_path)
    if [[ -n "$config_path" ]]; then
        echo ""
        echo "   Config location: $config_path"
    fi
}

# ----------------------------------------------------------------------------
# Log Management Functions
# ----------------------------------------------------------------------------

# Follow logs
follow_logs() {
    local log_path="$LOG_DIR/$LOG_FILE"
    
    echo "Following server logs (Ctrl+C to stop)..."
    echo ""
    
    # Create logs directory and file if they don't exist
    mkdir -p "$LOG_DIR"
    touch "$log_path"
    
    # Follow the log file
    tail -f "$log_path"
}

# ----------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------

main() {
    # Display header
    echo "🤖 Zen MCP Server"
    echo "================"
    
    # Get and display version
    local version=$(get_version)
    echo "Version: $version"
    echo ""
    
    # Check if venv exists
    if [[ ! -d "$VENV_PATH" ]]; then
        echo "Setting up Python environment for first time..."
    fi
    
    # Step 1: Docker cleanup
    cleanup_docker
    
    # Step 2: Find Python
    local python_cmd
    python_cmd=$(find_python) || exit 1
    
    # Step 3: Setup environment file
    setup_env_file || exit 1
    
    # Step 4: Source .env file
    if [[ -f .env ]]; then
        set -a
        source .env
        set +a
    fi
    
    # Step 5: Validate API keys
    validate_api_keys || exit 1
    
    # Step 6: Setup virtual environment
    local new_python_cmd
    new_python_cmd=$(setup_venv "$python_cmd")
    python_cmd="$new_python_cmd"
    
    # Step 7: Install dependencies
    install_dependencies "$python_cmd" || exit 1
    
    # Step 8: Get absolute server path
    local script_dir=$(get_script_dir)
    local server_path="$script_dir/server.py"
    
    # Step 9: Display setup instructions
    display_setup_instructions "$python_cmd" "$server_path"
    
    # Step 10: Check Claude integration
    check_claude_integration "$python_cmd" "$server_path"
    
    # Step 11: Display log information
    echo ""
    echo "Logs will be written to: $script_dir/$LOG_DIR/$LOG_FILE"
    echo ""
    
    # Step 12: Handle command line arguments
    if [[ "${1:-}" == "-f" ]] || [[ "${1:-}" == "--follow" ]]; then
        follow_logs
    else
        echo "To follow logs: ./run-server.sh -f"
        echo "To update: git pull, then run ./run-server.sh again"
        echo ""
        echo "Happy Clauding! 🎉"
    fi
}

# ----------------------------------------------------------------------------
# Script Entry Point
# ----------------------------------------------------------------------------

# Run main function with all arguments
main "$@"