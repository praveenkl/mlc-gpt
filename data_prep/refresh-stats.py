import json
import os
import requests
import sys
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy import text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class BattingPlayer(Base):
    __tablename__ = 'batting_statistics'
    ID = Column(Integer, primary_key=True, nullable=False)
    Name = Column(String)
    BattingStyle = Column(String)
    Role = Column(String)
    Team = Column(String)
    Matches = Column(Integer)
    InningsBatted = Column(Integer)
    RunsScored = Column(Integer)
    BallsFaced = Column(Integer)
    DotBallsPlayed = Column(Integer)
    FoursHit = Column(Integer)
    SixesHit = Column(Integer)
    NotOuts = Column(Integer)
    Fifties = Column(Integer)
    Centuries = Column(Integer)
    HighestScore = Column(String)
    BattingStrikeRate = Column(Float, nullable=True)
    BattingAverage = Column(Float, nullable=True)

class BowlingPlayer(Base):
    __tablename__ = 'bowling_statistics'
    ID = Column(Integer, primary_key=True, nullable=False)
    Name = Column(String)
    BowlingStyle = Column(String)
    Role = Column(String)
    Team = Column(String)
    Matches = Column(Integer)
    InningsBowled = Column(Integer)
    RunsGiven = Column(Integer)
    BallsBowled = Column(Integer)
    Overs = Column(Float, nullable=True)
    DotBallsBowled = Column(Integer)
    FoursGiven = Column(Integer)
    SixesGiven = Column(Integer)
    Wickets = Column(Integer)
    FourWickets = Column(Integer)
    FiveWickets = Column(Integer)
    TenWickets = Column(Integer)
    Maidens = Column(Integer)
    HighestWickets = Column(String)
    BowlingStrikeRate = Column(Float, nullable=True)
    BowlingAverage = Column(Float, nullable=True)
    EconomyRate = Column(Float, nullable=True)

class Match(Base):
    __tablename__ = 'match_details'
    ID = Column(Integer, primary_key=True, nullable=False)
    TeamAName = Column(String)
    TeamBName = Column(String)
    DateTime = Column(String)
    GroundName = Column(String)
    City = Column(String)
    TossInfo = Column(String)
    InningsOneSummary = Column(String)
    InningsTwoSummary = Column(String)
    WinTeamName = Column(String)
    ManOfTheMatchName = Column(String)
    Result = Column(String)

class Team(Base):
    __tablename__ = 'team_statistics'
    ID = Column(Integer, primary_key=True, nullable=False)
    Name = Column(String)
    FullName = Column(String)
    Matches = Column(Integer)
    Wins = Column(Integer)
    Loss = Column(Integer)
    Points = Column(Integer)
    NetRunRate = Column(Float)
    Image = Column(String)

def to_title_case(s):
    return ' '.join([word.capitalize() for word in s.split()])

def download_data():
    url_dict = {
        "Team": "https://splcms.blob.core.windows.net/cricketnest/client_1/mlc_match_center/Stats/Standings/1053-Standings.json",
        "Match": "https://splcms.blob.core.windows.net/cricketnest/client_1/mlc_match_center/Schedules/1053-Results.json",
        "Batting": "https://splcms.blob.core.windows.net/cricketnest/client_1/mlc_match_center/Stats/1053-MostRuns.json",
        "Bowling": "https://splcms.blob.core.windows.net/cricketnest/client_1/mlc_match_center/Stats/1053-MostWickets.json"
    }
    for key, url in url_dict.items():
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join("./raw", f"{key.lower()}.json"), "w") as file:
                file.write(response.text)
        else:
            print(f"Failed to download {key} data from {url}")
            sys.exit(1)

# Download latest statistics
download_data()

# Create a dictionary of team names and team full names
team_full_name_dict = {
    "SO": "Seattle Orcas",
    "TSK": "Texas Super Kings",
    "WTF": "Washington Freedom",
    "MINY": "Mumbai Indians New York",
    "SFU": "San Francisco Unicorns",
    "LAKR": "Los Angeles Knight Riders"
}
# team_full_name_dict = {
#     "SEA": "Seattle Orcas",
#     "TSK": "Texas Super Kings",
#     "WSH": "Washington Freedom",
#     "MI NY": "Mumbai Indians New York",
#     "SF": "San Francisco Unicorns",
#     "LAKR": "Los Angeles Knight Riders"
# }

# Database setup
if os.path.exists("../data/stats/mlc_stats_2024.db"):
    os.rename("../data/stats/mlc_stats_2024.db", "../data/stats/mlc_stats_2024.db.bak")
engine = create_engine("sqlite:///../data/stats/mlc_stats_2024.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Load and process batting statistics
with open("./raw/batting.json") as f:
    bat_data = json.load(f)
batting_players = []
for b in bat_data['CompetitionPlayerStats']:
    strike_rate = float(b['StrikeRate'])
    average = None if b['BattingAverage'] == "NA" else float(b['BattingAverage'])
    p = BattingPlayer(
        Name=to_title_case(b['PlayerName']),
        BattingStyle=b['BattingStyle'],
        Role=b['PlayerRole'],
        Team=b['TeamName'],
        Matches=b['Matches'],
        InningsBatted=b['Innings'],
        RunsScored=b['Runs'],
        BallsFaced=b['Balls'],
        DotBallsPlayed=b['DotBalls'],
        FoursHit=b['BdryFours'],
        SixesHit=b['BdrySixes'],
        NotOuts=b['NotOuts'],
        Fifties=b['Fifties'],
        Centuries=b['Centuries'],
        HighestScore=b['HighestScore'],
        BattingStrikeRate=strike_rate,
        BattingAverage=average
    )
    batting_players.append(p)
session.bulk_save_objects(batting_players)
session.commit()

# Load and process bowling statistics
with open("./raw/bowling.json") as f:
    bowl_data = json.load(f)
bowling_players = []
for b in bowl_data['CompetitionPlayerStats']:
    average = None if b['BowlingAverage'] == "NA" else float(b['BowlingAverage'])
    overs = float(b['Overs'])
    economy_rate = float(b['EconomyRate'])
    p = BowlingPlayer(
        Name=to_title_case(b['PlayerName']),
        BowlingStyle=b['BowlingStyle'],
        Role="",
        Team=b['TeamName'],
        Matches=b['Matches'],
        InningsBowled=b['Innings'],
        RunsGiven=b['Runs'],
        BallsBowled=b['Balls'],
        DotBallsBowled=b['DotBalls'],
        FoursGiven=b['BdryFours'],
        SixesGiven=b['BdrySixes'],
        Overs=overs,
        Wickets=b['Wickets'],
        FourWickets=b['FourWickets'],
        FiveWickets=b['FiveWickets'],
        TenWickets=b['TenWickets'],
        Maidens=b['Maidens'],
        HighestWickets=b['BestWickets'],
        BowlingStrikeRate=b['StrikeRate'],
        BowlingAverage=average,
        EconomyRate=economy_rate
    )
    bowling_players.append(p)
session.bulk_save_objects(bowling_players)
session.commit()

# Load and process match data
with open("./raw/match.json") as f:
    match_data = json.load(f)
matches = []
completed_match_info = {}
for m in match_data['CompetitionDeatails']:
    match = Match(
        TeamAName=m['TeamAName'],
        TeamBName=m['TeamBName'],
        DateTime=m['MatchDateTime'],
        GroundName=to_title_case(m['GroundName'].split(",")[0]),
        City=to_title_case(m['City']),
        TossInfo=m['TossInfo'],
        InningsOneSummary=m['InningsOneSummary'],
        InningsTwoSummary=m['InningsTwoSummary'],
        WinTeamName=m['WinTeamName'],
        ManOfTheMatchName="",
        Result=m['MatchResult']
    )
    matches.append(match)
    if len(m["MatchResult"]) > 0:
        team_A_fullname = m['TeamAName']
        team_B_fullname = m['TeamBName']
        # city = m['City'] # Ignoring city for now as it is always empty
        completed_match_info[m['MatchCode']] = (m["MatchOrder"], f'{team_A_fullname} vs {team_B_fullname}', m["MatchDate"], m["GroundName"], m['MatchResult'])
session.bulk_save_objects(matches)
session.commit()

# Write completed match info to a file in data/ directory. Create the directory if it doesn't exist.
with open("../data/match_reports/2024/completed_matches.json", "w") as f:
    json.dump(completed_match_info, f, indent=4)

# Load and process team statistics
with open("./raw/team.json") as f:
    team_data = json.load(f)
teams = []
for t in team_data['TournamentPoints']:
    team = Team(
        Name=t['TeamName'],
        FullName=team_full_name_dict[t['TeamName']],
        Matches=t['Matches'],
        Wins=t['Wins'],
        Loss=t['Loss'],
        Points=t['Points'],
        NetRunRate=t['NetRunRate'],
        Image=t['TeamLogo']
    )
    teams.append(team)
session.bulk_save_objects(teams)
session.commit()

print("Data inserted successfully.")

# Query examples
target_player_name = "N Pooran"
player = session.query(BattingPlayer).filter(BattingPlayer.Name == target_player_name).first()
if player:
    print(player.Name)
else:
    print("Player not found.")

target_team_name = "Mumbai Indians New York"
team = session.query(Team).filter(Team.FullName == target_team_name).first()
if team:
    print(team.Name, team.FullName, team.Matches, team.Wins, team.Loss, team.Points, team.NetRunRate)
else:
    print("Team not found.")

target_winning_team_name = "Los Angeles Knight Riders"
match = session.query(Match).filter(Match.WinTeamName == target_winning_team_name).first()
if match:
    print(match.Result)
else:
    print("Match not found.")

# Raw query example
# raw_query = text("SELECT WinTeamName, Result FROM matches WHERE TeamAName = 'SEA' and TeamBName = 'MI NY' AND City = 'Dallas'")
# raw_query = text("SELECT * FROM batting_players WHERE Name = 'Nicholas Pooran'")
# raw_query = text("SELECT * FROM batting_players WHERE id < 2")
# raw_query = text("SELECT Name, BattingStrikeRate FROM batting_players ORDER BY BattingStrikeRate ASC LIMIT 1")
# raw_query = text("SELECT Name, BattingStrikeRate FROM batting_players ORDER BY RunsScored DESC LIMIT 1")
# raw_query = text("SELECT Name, BowlingAverage FROM bowling_players ORDER BY BowlingAverage ASC LIMIT 1")
raw_query = text("SELECT TeamAName, TeamBName FROM match_details WHERE GroundName = 'Church Street Park'")
result = session.execute(raw_query)
for row in result:
    print(row)
