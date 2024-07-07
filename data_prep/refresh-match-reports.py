import json
import requests
import os
from openai import OpenAI

def create_match_summary_json(match_summary, ball_info):
    match_result = {
        "TeamAName": match_summary["MatchInfo"][0]["TeamAName"],
        "TeamBName": match_summary["MatchInfo"][0]["TeamBName"],
        "Venue": match_summary["MatchInfo"][0]["GroundName"],
        "City": match_summary["MatchInfo"][0]["City"],
        "TossInfo": match_summary["MatchInfo"][0]["TossInfo"],
        "WinTeamName": match_summary["MatchInfo"][0]["WinTeamName"],
        "MatchResult": match_summary["MatchInfo"][0]["MatchResult"],
        "ManOfTheMatch": match_summary["MatchInfo"][0]["ManOfTheMatch"],
        "MOMBattingScore": match_summary["MatchInfo"][0]["MOMBattingScore"],
        "MOMBowlingScore": match_summary["MatchInfo"][0]["MOMBowlingScore"],
        "InningsOneSummary": match_summary["MatchInfo"][0]["InningsOneSummary"],
        "InningsTwoSummary": match_summary["MatchInfo"][0]["InningsTwoSummary"]
    }
    team_codes = {}
    team_codes[match_summary["MatchInfo"][0]["TeamACode"]] = match_summary["MatchInfo"][0]["TeamAName"]
    team_codes[match_summary["MatchInfo"][0]["TeamBCode"]] = match_summary["MatchInfo"][0]["TeamBName"]

    batting_to_bowling_team_map = {}
    batting_to_bowling_team_map[match_summary["MatchInfo"][0]["TeamACode"]] = match_summary["MatchInfo"][0]["TeamBCode"]
    batting_to_bowling_team_map[match_summary["MatchInfo"][0]["TeamBCode"]] = match_summary["MatchInfo"][0]["TeamACode"]

    batting_highlights = []
    for player in match_summary["BattingSummary"]:
        player_team_name = team_codes[player["BattingTeamID"]]
        batting_highlights.append({
            "PlayerName": player["PlayerName"].strip(" *"),
            "PlayerTeamName": player_team_name,
            "Runs": player["Runs"],
            "Balls": player["Balls"],
            "RunText": player["RunText"],
            "InningsNo": int(player["InningsNo"]),
        })
    
    bowling_highlights = []
    for player in match_summary["BowlingSummary"]:
        bowling_team_id = batting_to_bowling_team_map[player["BattingTeamID"]]
        player_team_name = team_codes[bowling_team_id]
        bowling_highlights.append({
            "PlayerName": player["PlayerName"],
            "PlayerTeamName": player_team_name,
            "Overs": player["Overs"],
            "RunsGiven": player["Runs"],
            "Wickets": player["Wickets"],
            "WicketText": player["WicketText"],
            "InningsNo": int(player["InningsNo"]),
        })

    team_milestones = []
    player_milestones = []
    for ball in ball_info["BallInfo"]:
        if ball["IsTeamMilestone"] == 1:
            team_milestones.append({
                "OverNumber": ball["OverNo"],
                "BallNumber": ball["BallNO"],
                "OverValue": ball["OverValue"],
                "TeamMilestone": ball["TeamMilestone"],
                "CommentryText": ball["CommentryText"]
            })
        if ball["IsPlayerMilestone"] == 1:
            player_milestones.append({
                "OverNumber": ball["OverNo"],
                "BallNumber": ball["BallNO"],
                "OverValue": ball["OverValue"],
                "PlayerMilestone": ball["PlayerMilestone"],
                "CommentryText": ball["CommentryText"]
            })
    
    # Combine match_result, batting_highlights, bowling_highlights, team_milestones, player_milestones into a single json object
    match_summary = {
        "MatchResult": match_result,
        "BattingHighlights": batting_highlights,
        "BowlingHighlights": bowling_highlights,
        "TeamMilestones": team_milestones,
        "PlayerMilestones": player_milestones
    }
    match_summary_json = json.dumps(match_summary, indent=4)
    return match_summary_json

match_reports_dir = "../data/match_reports/2024"

def get_match_summary_objects():
    match_summary_objects = {}
    match_result_url_fmt = "https://splcms.blob.core.windows.net/cricketnest/client_1/mlc_match_center/Matches/{match_id}/GetMatchSummary.json"
    ball_info_url_fmt = "https://splcms.blob.core.windows.net/cricketnest/client_1/mlc_match_center/Matches/{match_id}/BallInfo.json"

    with open(f'{match_reports_dir}/completed_matches.json') as f:
        completed_matches = json.load(f)
    match_ids = list(completed_matches.keys())

    for match_id in match_ids:
        if os.path.exists(f'{match_reports_dir}/{match_id}_report.txt'):
            continue
        match_result_url = match_result_url_fmt.format(match_id=match_id)
        ball_info_url = ball_info_url_fmt.format(match_id=match_id)
        match_result_json = requests.get(match_result_url).text
        ball_info_json = requests.get(ball_info_url).text
        with open(f'{match_reports_dir}/{match_id}_match_result.json', 'w') as f:
            f.write(match_result_json)
        with open(f'{match_reports_dir}/{match_id}_ball_info.json', 'w') as f:
            f.write(ball_info_json)
        match_summary_json = create_match_summary_json(json.loads(match_result_json), json.loads(ball_info_json))
        match_summary_objects[match_id] = match_summary_json

    for match_id, match_summary_json in match_summary_objects.items():
        with open(f'{match_reports_dir}/{match_id}_summary.json', 'w') as f:
            f.write(match_summary_json)

    return match_summary_objects

def generate_match_report(match_summary_json):
    system_prompt = '''
You are a helpful assistant highly skilled at generating a T20 cricket match report. You will be provided with a json file which includes the details about a completed T20 cricket match from the major league cricket (MLC) tournament. You will use these details to create the match report.

The JSON file provided to you has the following sections.
1. MatchResult — This section includes details about the teams that played the match, the venue, toss, the scores for both the innings, the final result and the man of the match.
2. Batting Highlights — This section includes details about notable batting performances from both the teams.
3. Bowling Highlights — This section includes details about notable bowling performances from both the teams.
4. Team Milestones — This section includes details about some team milestones including the exact moment in the match these milestones occurred and the associated commentary text.
5. Player Milestones — This section includes details about some player milestones including the exact moment in the match these milestones occurred and the associated commentary text.

Generate your report in a narrative style and do not divide it into sections.
''' 
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": match_summary_json}
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    try:
        match_summary_objects = get_match_summary_objects()
        print("Match summary objects created successfully")
        for match_id, match_summary_json in match_summary_objects.items():
            match_report = generate_match_report(match_summary_json)
            with open(f'{match_reports_dir}/{match_id}_report.txt', 'w') as f:
                f.write(match_report)
            print(f"Match report for match {match_id} created successfully")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Update the schedule.json file by removing the completed matches
    with open(f'{match_reports_dir}/schedule.json') as f:
        schedule = json.load(f)
    for match_id in match_summary_objects.keys():
        try:
            del schedule[match_id]
        except KeyError:
            pass
    with open(f'{match_reports_dir}/schedule.json', 'w') as f:
        json.dump(schedule, f, indent=4)
    print("Schedule updated successfully")
