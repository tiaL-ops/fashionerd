from flask import Flask, render_template
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Function to scrape Pinterest fashion trends
def scrape_pinterest_fashion():
    url = "https://www.pinterest.com/search/pins/?q=fashion%20trends"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')


        images = soup.find_all('img', {'src': True})
        
       
        fashion_trends = []
        for img in images:
            img_url = img['src']
            description = img.get('alt', 'No description')
            fashion_trends.append({'url': img_url, 'description': description})
        
        return fashion_trends
    else:
        return []



@app.route('/')
def home():
    
    fashion_data = scrape_pinterest_fashion()
    return render_template('index.html', fashion_data=fashion_data)


if __name__ == '__main__':
    app.run(debug=True)
