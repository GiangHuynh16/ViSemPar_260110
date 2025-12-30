# Request Permission for Server Directory

**Server**: `islab-server2`
**User**: `islabworker2`
**Target Directory**: `/mnt/nghiepth/giangha/visempar`

---

## Option 1: Request Admin to Grant Permission

### Commands for Admin (requires sudo password):

```bash
# Grant ownership to your user
sudo chown -R islabworker2:islabworker2 /mnt/nghiepth/giangha/visempar

# Grant read/write/execute permissions
sudo chmod -R 755 /mnt/nghiepth/giangha/visempar

# Verify permissions
ls -la /mnt/nghiepth/giangha/visempar
```

### Expected output after permission granted:
```
drwxr-xr-x 2 islabworker2 islabworker2 4096 Dec 30 10:00 visempar
```

---

## Option 2: You Run Commands with Your Password

If you have sudo access with your password:

```bash
# SSH to server
ssh islabworker2@islab-server2

# Create directory if not exists
sudo mkdir -p /mnt/nghiepth/giangha/visempar

# Grant yourself ownership
sudo chown -R islabworker2:islabworker2 /mnt/nghiepth/giangha/visempar

# Grant permissions
sudo chmod -R 755 /mnt/nghiepth/giangha/visempar

# Test permission
touch /mnt/nghiepth/giangha/visempar/test.txt
rm /mnt/nghiepth/giangha/visempar/test.txt

# If successful, you now have full access!
```

---

## Option 3: Use Existing Directory with Permission

Check if you have access to other directories:

```bash
# Check your home directory (you always have access here)
ls -la ~/
cd ~
mkdir -p visempar
cd visempar

# OR check other shared directories
ls -la /mnt/nghiepth/giangha/
# Look for directories owned by islabworker2

# OR check if visempar already exists with different permissions
ls -la /mnt/nghiepth/giangha/ | grep visempar
```

---

## Option 4: Create in Home Directory as Fallback

If you cannot get permission for `/mnt/nghiepth/giangha/visempar`:

```bash
# Use your home directory instead
cd ~
mkdir -p visempar
cd visempar

# Clone repo here
git clone git@github.com:GiangHuynh16/ViSemPar_new1.git

# Full path will be: /home/islabworker2/visempar/ViSemPar_new1
```

---

## Verify Permission Status

Run these commands to check current status:

```bash
# Check if directory exists
ls -la /mnt/nghiepth/giangha/ | grep visempar

# Check ownership and permissions
stat /mnt/nghiepth/giangha/visempar

# Try to create test file
touch /mnt/nghiepth/giangha/visempar/permission_test.txt

# If successful:
echo "✓ You have write permission"
rm /mnt/nghiepth/giangha/visempar/permission_test.txt

# If failed:
echo "✗ Need to request permission"
```

---

## Quick Test Script

Save this as `test_permission.sh` and run on server:

```bash
#!/bin/bash

TARGET_DIR="/mnt/nghiepth/giangha/visempar"

echo "Testing permission for: $TARGET_DIR"
echo ""

# Test 1: Check if directory exists
if [ -d "$TARGET_DIR" ]; then
    echo "✓ Directory exists"
else
    echo "✗ Directory does not exist"
    echo "  Need to create it (requires permission)"
    exit 1
fi

# Test 2: Check ownership
OWNER=$(stat -c '%U' "$TARGET_DIR" 2>/dev/null || stat -f '%Su' "$TARGET_DIR" 2>/dev/null)
echo "  Owner: $OWNER"

if [ "$OWNER" = "islabworker2" ]; then
    echo "  ✓ You own this directory"
else
    echo "  ✗ You do NOT own this directory (owned by $OWNER)"
fi

# Test 3: Test write permission
TEST_FILE="$TARGET_DIR/.permission_test_$$"
if touch "$TEST_FILE" 2>/dev/null; then
    echo "✓ Write permission: YES"
    rm -f "$TEST_FILE"
else
    echo "✗ Write permission: NO"
    echo ""
    echo "Solutions:"
    echo "1. Ask admin to run:"
    echo "   sudo chown -R islabworker2:islabworker2 $TARGET_DIR"
    echo "   sudo chmod -R 755 $TARGET_DIR"
    echo ""
    echo "2. Or use home directory instead:"
    echo "   mkdir -p ~/visempar"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ ALL TESTS PASSED"
echo "You have full access to: $TARGET_DIR"
echo "=========================================="
```

---

## Summary

### If you have sudo password:
```bash
sudo chown -R islabworker2:islabworker2 /mnt/nghiepth/giangha/visempar
sudo chmod -R 755 /mnt/nghiepth/giangha/visempar
```

### If you DON'T have sudo access:
Contact server admin and ask them to run the commands above.

### Fallback:
```bash
# Use home directory
cd ~
mkdir -p visempar
cd visempar
git clone git@github.com:GiangHuynh16/ViSemPar_new1.git
```

After getting permission, proceed with [SETUP_NEW_SERVER.md](SETUP_NEW_SERVER.md)
