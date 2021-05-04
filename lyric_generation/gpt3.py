import os
import openai
from dotenv import load_dotenv

load_dotenv(dotenv_path='../.env')  # take environment variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="ada",
  prompt="Write Love songs\n\n1.\nWe were both young when I first saw you\nI close my eyes and the flashback starts\nI'm standing there\nOn a balcony in summer air\nSee the lights, see the party, the ball gowns\nSee you make your way through the crowd\nAnd say, \"Hello\"\nLittle did I know\nThat you were Romeo, you were throwing pebbles\nAnd my daddy said, \"Stay away from Juliet\"\nAnd I was crying on the staircase\nBegging you, \"Please don't go,\" and I said\nRomeo, take me somewhere we can be alone\nI'll be waiting, all there's left to do is run\nYou'll be the prince and I'll be the princess\nIt's a love story, baby, just say, \"Yes\"\nSo I sneak out to the garden to see you\nWe keep quiet 'cause we're dead if they knew\nSo close your eyes\nEscape this town for a little while, oh, oh\n'Cause you were Romeo, I was a scarlet letter\nAnd my daddy said, \"Stay away from Juliet\"\nBut you were everything to me\nI was begging you, \"Please don't go,\" and I said\nRomeo, take me somewhere we can be alone\nI'll be waiting, all there's left to do is run\nYou'll be the prince and I'll be the princess\nIt's a love story, baby, just say, \"Yes\"\nRomeo, save me, they're trying to tell me how to feel\nThis love is difficult, but it's real\nDon't be afraid, we'll make it out of this mess\nIt's a love story, baby, just say, \"Yes\"\nOh, oh\nAnd I got tired of waiting\nWondering if you were ever coming around\nMy faith in you was fading\nWhen I met you on the outskirts of town\nAnd I said, \"Romeo, save me, I've been feeling so alone\nI keep waiting for you, but you never come\nIs this in my head? I don't know what to think\"\nHe knelt to the ground and pulled out a ring, and said\n\"Marry me, Juliet, you'll never have to be alone\nI love you and that's all I really know\nI talked to your dad, go pick out a white dress\nIt's a love story, baby, just say, 'Yes'\"\nOh, oh, oh\nOh, oh, oh\n'Cause we were both young when I first saw you\n####\n2.\nWell, it's a sad picture, the final blow hits you\nSomebody else gets what you wanted again and\nYou know it's all the same, another time and place\nRepeating history and you're getting sick of it\nBut I believe in whatever you do\nAnd I'll do anything to see it through\nBecause these things will change\nCan you feel it now?\nThese walls that they put up to hold us back will fall down\nThis revolution, the time will come\nFor us to finally win\nAnd we'll sing hallelujah, we'll sing hallelujah\nOh, oh\nSo we've been outnumbered, raided, and now cornered\nIt's hard to fight when the fight ain't fair\nWe're getting stronger now, find things they never found\nThey might be bigger but we're faster and never scared\nYou can walk away, say we don't need this\nBut there's something in your eyes says we can beat this\n'Cause these things will change\nCan you feel it now?\nThese walls that they put up to hold us back will fall down\nThis revolution, the time will come\nFor us to finally win\nAnd we'll sing hallelujah, we'll sing hallelujah\nOh, oh\nTonight, we'll stand, get off our knees\nFight for what we've worked for all these years\nAnd the battle was long, it's the fight of our lives\nBut we'll stand up champions tonight\nIt was the night things changed\nCan you see it now?\nWhen the walls that they put up to hold us back fell down\nIt's a revolution, throw your hands up\n'Cause we never gave in\nAnd we'll sing hallelujah, we sang hallelujah\nHallelujah\n####\n3.\nWe needed that gun\nIt's too bad, what is this life we live? Well, it's a lie\nAnd so many see us in the dark tonight\nIT'S GONE!So much pain is brought on by the sins we commit\nNow I don't need you to tell me I'm not enough\nI don't even know who I am anymore\nMay God forgive me if I've offended\nBut let me tell you the truth, I can't change how things are\n(Hallelujah)But it's a revolution, throw your hands up\n'Cause we never gave in\nAnd we'll sing hallelujah, we sang hallelujah\nOh, oh",
  temperature=0.9,
  max_tokens=900,
  top_p=1,
  frequency_penalty=0.2,
  presence_penalty=0.82
)

print(response)