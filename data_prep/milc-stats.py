import csv
import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class BattingPlayer(Base):
    __tablename__ = 'batting_statistics'
    ID = Column(Integer, primary_key=True, nullable=False)
    Name = Column(String)
    BattingStyle = Column(String, nullable=True)
    Role = Column(String, nullable=True)
    Team = Column(String)
    Matches = Column(Integer)
    InningsBatted = Column(Integer)
    RunsScored = Column(Integer)
    BallsFaced = Column(Integer, nullable=True)
    DotBallsPlayed = Column(Integer, nullable=True)
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
    BowlingStyle = Column(String, nullable=True)
    Role = Column(String, nullable=True)
    Team = Column(String)
    Matches = Column(Integer)
    InningsBowled = Column(Integer)
    RunsGiven = Column(Integer)
    BallsBowled = Column(Integer, nullable=True)
    Overs = Column(Float, nullable=True)
    DotBallsBowled = Column(Integer)
    FoursGiven = Column(Integer, nullable=True)
    SixesGiven = Column(Integer, nullable=True)
    Wickets = Column(Integer)
    FourWickets = Column(Integer)
    FiveWickets = Column(Integer)
    TenWickets = Column(Integer, nullable=True)
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
    GroundName = Column(String, nullable=True)
    City = Column(String, nullable=True)
    TossInfo = Column(String, nullable=True)
    InningsOneSummary = Column(String)
    InningsTwoSummary = Column(String)
    WinTeamName = Column(String, nullable=True)
    ManOfTheMatchName = Column(String, nullable=True)
    Result = Column(String)

class Team(Base):
    __tablename__ = 'team_statistics'
    ID = Column(Integer, primary_key=True, nullable=False)
    Name = Column(String)
    FullName = Column(String)
    Matches = Column(Integer)
    Wins = Column(Integer)
    Loss = Column(Integer)
    Points = Column(Float)
    NetRunRate = Column(Float)
    Image = Column(String, nullable=True)

def to_title_case(s):
    return ' '.join([word.capitalize() for word in s.split()])

# Database setup
if os.path.exists("./milc_stats_2024.db"):
    os.rename("./milc_stats_2024.db", "./milc_stats_2024.db.bak")
engine = create_engine("sqlite:///./milc_stats_2024.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Load and process batting statistics
with open("./milc_raw/batting.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    batting_players = []
    for row in reader:
        if not row['Player'].strip():
            continue  # Skip rows without player names
        p = BattingPlayer(
            Name=to_title_case(row['Player'].strip()),
            BattingStyle=None,  # Not available
            Role=None,          # Not available
            Team=row['Team'].strip(),
            Matches=int(row['Mat']),
            InningsBatted=int(row['Inns']),
            RunsScored=int(row['Runs']),
            BallsFaced=None,    # Not available
            DotBallsPlayed=None,  # Not available
            FoursHit=int(row["4's"]),
            SixesHit=int(row["6's"]),
            NotOuts=int(row['NO']),
            Fifties=int(row["50's"]),
            Centuries=int(row["100's"]),
            HighestScore=row['HS'].strip(),
            BattingStrikeRate=float(row['SR']),
            BattingAverage=None if (row['Avg'] == 'NA' or row['Avg'] == '--') else float(row['Avg'])
        )
        batting_players.append(p)
session.bulk_save_objects(batting_players)
session.commit()

# Load and process bowling statistics
with open("./milc_raw/bowling.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    bowling_players = []
    for row in reader:
        if not row['Player'].strip():
            continue  # Skip rows without player names
        overs = float(row['Overs'])
        whole_overs = int(overs)
        partial_overs = overs - whole_overs
        balls_in_partial_over = int(partial_overs * 10)
        balls_bowled = whole_overs * 6 + balls_in_partial_over
        p = BowlingPlayer(
            Name=to_title_case(row['Player'].strip()),
            BowlingStyle=None,  # Not available
            Role=None,          # Not available
            Team=row['Team'].strip(),
            Matches=int(row['Mat']),
            InningsBowled=int(row['Inns']),
            RunsGiven=int(row['Runs']),
            BallsBowled=balls_bowled,
            Overs=overs,
            DotBallsBowled=int(row['dots']),
            FoursGiven=None,    # Not available
            SixesGiven=None,    # Not available
            Wickets=int(row['Wkts']),
            FourWickets=int(row['4w']),
            FiveWickets=int(row['5w']),
            TenWickets=0,       # Not available, assume 0
            Maidens=int(row['Mdns']),
            HighestWickets=row['BBF'].strip(),
            BowlingStrikeRate=float(row['SR']),
            BowlingAverage=None if row['Avg'] == 'NA' else float(row['Avg']),
            EconomyRate=float(row['Econ'])
        )
        bowling_players.append(p)
session.bulk_save_objects(bowling_players)
session.commit()

# Load and process match data
with open("./milc_raw/match.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    matches = []
    for row in reader:
        # Parse date (assuming MM/DD/YYYY format)
        date_str = row['DATE'].strip()
        original_format = "%m/%d/%Y"
        datetime_obj = datetime.strptime(date_str, original_format)
        date_formatted = datetime_obj.strftime("%Y-%m-%d")

        # Extract winning team from RESULT
        result = row['RESULT'].strip()
        win_team_name = None
        if "won by" in result:
            win_team_name = result.split(" won by")[0]

        # Split SCORE SUMMARY into Innings One and Two
        score_summary = row['SCORE SUMMARY'].strip()
        team_one = row['Team ONE'].strip()
        team_two = row['TEAM TWO'].strip()
        innings = score_summary.split(team_two + ":")
        innings_one_summary = innings[0].strip()
        innings_two_summary = team_two + ":" + innings[1].strip() if len(innings) > 1 else ""

        match = Match(
            TeamAName=team_one,
            TeamBName=team_two,
            DateTime=date_formatted,
            GroundName=None,      # Not available
            City=None,            # Not available
            TossInfo=None,        # Not available
            InningsOneSummary=innings_one_summary,
            InningsTwoSummary=innings_two_summary,
            WinTeamName=win_team_name,
            ManOfTheMatchName=None,  # Not available
            Result=result
        )
        matches.append(match)
session.bulk_save_objects(matches)
session.commit()

# Load and process team statistics
with open("./milc_raw/team.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    teams = []
    for row in reader:
        net_rr = row['NET RR'].replace('%', '').strip()
        net_rr = float(net_rr) if net_rr else 0.0
        team = Team(
            Name=row['TEAM'].strip(),
            FullName=row['TEAM'].strip(),  # Assuming FullName is same as Name
            Matches=int(row['MAT']),
            Wins=int(row['WON']),
            Loss=int(row['LOST']),
            Points=float(row['PTS']),
            NetRunRate=net_rr,
            Image=None  # Not available
        )
        teams.append(team)
session.bulk_save_objects(teams)
session.commit()

print("Data inserted successfully.")

# Query examples (same as before)
target_player_name = "Sujith Gowda"
player = session.query(BattingPlayer).filter(BattingPlayer.Name == target_player_name).first()
if player:
    print(player.Name)
else:
    print("Player not found.")

target_team_name = "East Bay Blazers"
team = session.query(Team).filter(Team.Name == target_team_name).first()
if team:
    print(team.Name, team.Matches, team.Wins, team.Loss, team.Points, team.NetRunRate)
else:
    print("Team not found.")

target_winning_team_name = "Chicago Kingsmen"
match = session.query(Match).filter(Match.TeamAName == target_winning_team_name).first()
if match:
    print(match.Result)
else:
    print("Match not found.")
