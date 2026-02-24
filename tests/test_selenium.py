import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "http://localhost:5000"

@pytest.fixture(scope="module")
def driver():
    """Setup Chrome WebDriver."""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new')  # Run headless for CI/quiet testing
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    
    # Initialize driver (Selenium 4.6+ automatically handles driver management)
    driver = webdriver.Chrome(options=options)
    
    yield driver
    
    # Teardown
    driver.quit()

def test_landing_page_selenium(driver):
    """Test landing page loads correctly using Selenium."""
    driver.get(BASE_URL)
    
    # Check Title
    assert "Truth Guard" in driver.title
    
    # Wait for Hero Title and check text
    hero_title = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "h1.hero-title"))
    )
    assert "Can You Tell What's Real Anymore?" in hero_title.text

def test_detector_page_selenium(driver):
    """Test detector page loads and core elements are present using Selenium."""
    driver.get(f"{BASE_URL}/detector")
    
    # Wait for Navbar brand
    brand = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".navbar-brand-custom"))
    )
    assert "TRUTHGUARD" in brand.text
    
    # Check Input Area
    input_area = driver.find_element(By.ID, "newsInput")
    assert input_area.is_displayed()
    
    # Check Analyze Button
    analyze_btn = driver.find_element(By.ID, "analyzeBtn")
    assert analyze_btn.is_displayed()
    
    # Check Mode Toggles
    ml_btn = driver.find_element(By.ID, "mlModeBtn")
    ai_btn = driver.find_element(By.ID, "aiModeBtn")
    assert ml_btn.is_displayed()
    assert ai_btn.is_displayed()
