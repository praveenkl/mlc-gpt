import os
import time
from openai import OpenAI

def generate_match_report(match_scorecard_csv):
    system_prompt = '''
You are a helpful assistant highly skilled at generating a T20 cricket match report. You will be provided with a CSV file which is a copy of the scorecard of a completed T20 cricket match from the minor league cricket (MiLC) tournament. You will use the scorecard to create the match report.

Generate your report in a narrative style and do not divide it into sections.
''' 
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": match_scorecard_csv}
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    # Read the csv files in match_csv_files directory and generate match reports
    # Save the match reports in the match_reports directory
    # Create the directories if they don't exist
    match_csv_files = os.listdir("match_csv_files")
    if not os.path.exists("match_reports"):
        os.makedirs("match_reports")
    for match_csv_file in match_csv_files:
        print(f"Generating match report for {match_csv_file}")
        with open(f"match_csv_files/{match_csv_file}", "r", errors="replace") as f:
            match_scorecard_csv = f.read()
        match_report = generate_match_report(match_scorecard_csv)
        with open(f"match_reports/{match_csv_file.replace('.csv', '.txt')}", "w") as f:
            f.write(match_report)
        print(f"Generated match report for {match_csv_file}")
        # Sleep for 2 seconds to avoid rate limiting
        time.sleep(2)
