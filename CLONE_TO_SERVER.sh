#!/bin/bash
# Clone repository to server directory
# Run this script ON THE SERVER (islab-server2)

set -e

TARGET_DIR="/mnt/nghiepth/giangha/visempar"
REPO_URL="git@github.com:GiangHuynh16/ViSemPar_new1.git"

echo "=========================================="
echo "CLONE REPOSITORY TO SERVER"
echo "=========================================="
echo ""
echo "Target directory: $TARGET_DIR"
echo "Repository: $REPO_URL"
echo ""

# Step 1: Check if directory exists and has permission
echo "Step 1: Checking directory permission..."
if [ ! -d "$TARGET_DIR" ]; then
    echo "  Directory does not exist. Creating..."
    if mkdir -p "$TARGET_DIR" 2>/dev/null; then
        echo "  ✓ Created directory"
    else
        echo "  ✗ Cannot create directory (need sudo)"
        echo ""
        echo "Please run with sudo:"
        echo "  sudo mkdir -p $TARGET_DIR"
        echo "  sudo chown -R $USER:$USER $TARGET_DIR"
        echo ""
        echo "Or use home directory:"
        echo "  mkdir -p ~/visempar && cd ~/visempar"
        exit 1
    fi
fi

# Test write permission
if touch "$TARGET_DIR/.test_$$" 2>/dev/null; then
    rm -f "$TARGET_DIR/.test_$$"
    echo "  ✓ Have write permission"
else
    echo "  ✗ No write permission"
    echo ""
    echo "Need to fix permission first:"
    echo "  sudo chown -R $USER:$USER $TARGET_DIR"
    echo "  sudo chmod -R 755 $TARGET_DIR"
    echo ""
    exit 1
fi
echo ""

# Step 2: Navigate to directory
echo "Step 2: Navigating to directory..."
cd "$TARGET_DIR"
echo "  Current directory: $(pwd)"
echo ""

# Step 3: Check if repo already exists
echo "Step 3: Checking if repository exists..."
if [ -d "ViSemPar_new1" ]; then
    echo "  Repository already exists!"
    echo ""
    read -p "  Pull latest changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd ViSemPar_new1
        echo "  Pulling latest changes..."
        git pull origin main
        echo "  ✓ Repository updated"
    else
        echo "  Skipping update"
    fi
else
    echo "  Repository does not exist. Cloning..."

    # Step 4: Check SSH key
    echo ""
    echo "Step 4: Checking SSH key for GitHub..."
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "  ✓ SSH key configured"
    else
        echo "  ✗ SSH key not configured"
        echo ""
        echo "Please setup SSH key first:"
        echo ""
        echo "1. Generate SSH key:"
        echo "   ssh-keygen -t ed25519 -C \"your_email@example.com\""
        echo ""
        echo "2. Copy public key:"
        echo "   cat ~/.ssh/id_ed25519.pub"
        echo ""
        echo "3. Add to GitHub:"
        echo "   https://github.com/settings/keys"
        echo ""
        echo "4. Test connection:"
        echo "   ssh -T git@github.com"
        echo ""
        exit 1
    fi

    # Step 5: Clone repository
    echo ""
    echo "Step 5: Cloning repository..."
    git clone "$REPO_URL"
    echo "  ✓ Repository cloned"

    cd ViSemPar_new1
fi

echo ""
echo "=========================================="
echo "SUCCESS!"
echo "=========================================="
echo ""
echo "Repository location: $(pwd)"
echo ""
echo "Next steps:"
echo ""
echo "1. Run setup script:"
echo "   bash QUICK_START_NEW_SERVER.sh"
echo ""
echo "2. Or verify and start training:"
echo "   bash VERIFY_AND_START.sh"
echo ""
