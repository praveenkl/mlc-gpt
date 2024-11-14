import requests
from bs4 import BeautifulSoup
import json
import os

def main():
    # URL of the webpage to scrape
    url = 'https://cricclubs.com/MiLC/listMatches.do?league=0&year=2024&clubId=18036'
    
    # Create a session
    session = requests.Session()

    # Get the webpage content
    response = session.get(url)
    response.raise_for_status()  # Check for request errors

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'lxml')

    # Find the table with match data
    match_table = soup.find('table', id='schedule-table1')
    if not match_table:
        print("Match table not found.")
        return

    # Dictionary to hold match summaries
    match_summaries = {}

    # Ensure the directory for CSV files exists
    csv_dir = 'match_csv_files'
    os.makedirs(csv_dir, exist_ok=True)

    # Iterate over each row in the table body
    for idx, row in enumerate(match_table.find('tbody').find_all('tr'), start=1):
        # Extracting relevant columns
        cols = row.find_all('td')
        if not cols:
            continue

        # Match ID and Club ID
        scorecard_link = cols[6].find('a', href=True)
        if not scorecard_link:
            continue

        href = scorecard_link['href']
        # Example href: '/MiLC/viewScorecard.do?matchId=694&clubId=18036'
        params = href.split('?')[-1]
        param_dict = dict(param.split('=') for param in params.split('&'))
        match_id = param_dict.get('matchId')
        club_id = param_dict.get('clubId')

        if not match_id or not club_id:
            continue

        # Modify URL to download scorecard as CSV
        excel_url = f'https://cricclubs.com/MiLC/viewScorecardExcel.do?matchId={match_id}&clubId={club_id}'

        # Download the Excel file and save as CSV
        excel_response = session.get(excel_url)
        excel_response.raise_for_status()

        csv_path = os.path.join(csv_dir, f'{match_id}.csv')
        with open(csv_path, 'wb') as f:
            f.write(excel_response.content)
        print(f'Downloaded CSV for match {match_id}')

        # Extracting other details
        match_index = int(cols[0].get_text(strip=True))
        date = cols[2].get_text(strip=True)
        team_one = cols[3].get_text(strip=True)
        team_two = cols[4].get_text(strip=True)
        result = cols[5].get_text(strip=True).upper()

        # Venue is not provided, so we'll set it as unknown
        venue = 'Unknown Venue'

        # Formatting the date and match info
        formatted_date = date.replace("/", " ")
        teams = f"{team_one} vs {team_two}"

        # Add to match summaries dictionary
        match_summaries[match_id] = [
            match_index,
            teams,
            formatted_date.upper(),
            venue,
            result
        ]

    # Save match summaries to a JSON file
    with open('match_summaries.json', 'w') as json_file:
        json.dump(match_summaries, json_file, indent=4)

    print('Match summaries saved to match_summaries.json')

if __name__ == '__main__':
    main()
