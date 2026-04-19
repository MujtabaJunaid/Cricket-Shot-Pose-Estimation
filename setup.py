import argparse
import logging
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_project():
    logger.info("Initializing Cricket Pose Classification project...")
    
    directories = [
        'data/raw/drive',
        'data/raw/pull',
        'data/raw/cut',
        'data/raw/defense',
        'data/raw/backfoot_play',
        'data/raw/cover_drive',
        'data/raw/off_drive',
        'data/raw/on_drive',
        'data/raw/lofted_shot',
        'data/processed',
        'models/checkpoints',
        'logs',
        'uploads'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    logger.info("Project initialized successfully!")
    logger.info("Please add training data to data/raw/{shot_class}/ directories")

def setup_backend():
    logger.info("Setting up backend environment...")
    import subprocess
    
    try:
        subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
        logger.info("Backend dependencies installed successfully!")
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False
    
    return True

def setup_frontend():
    logger.info("Setting up frontend environment...")
    import subprocess
    
    try:
        subprocess.run(['npm', 'install'], cwd='frontend', check=True)
        logger.info("Frontend dependencies installed successfully!")
    except Exception as e:
        logger.error(f"Failed to install frontend dependencies: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Cricket Pose Classification System Setup'
    )
    parser.add_argument(
        'command',
        choices=['init', 'setup', 'setup-backend', 'setup-frontend'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    if args.command == 'init':
        init_project()
    elif args.command == 'setup':
        init_project()
        setup_backend()
        setup_frontend()
        logger.info("Full setup completed!")
    elif args.command == 'setup-backend':
        setup_backend()
    elif args.command == 'setup-frontend':
        setup_frontend()

if __name__ == '__main__':
    main()
