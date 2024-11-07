---
layout: post
title:  "Data Analysis of Magic: The Gathering Cards"
date:   2024-11-03 11:35:15 +0200
categories: jekyll update
---
# Overview
Since I recently got back into Magic: The Gathering (MTG) I wanted to have a look at the data and statistics behind the available cards. MTG is a trading card game by "Wizards of the Coast" (WotC) released in 1993.  
Luckily Scryfall has a huge dataset of all the cards from their DB available [here](https://scryfall.com/docs/api/bulk-data). The data is available as a .json file which I first loaded with Python and converted to a pandas dataframe to further process and clean it.  
The dataset has around 100.000 entries corresponding to 100.000 different cards.

## Colors
First step is to look at the color identities of the cards. As expected they are pretty well distributed and all possible color identities are well represented.  
For simplicity I decided to assign the first color of multi-color cards as the primary identity. This means a card which has the [Green, Red] color identity will be counted as Green.

<div style="text-align: center">
    <img src="{{ '/assets/img/color_identity_pie_chart.png' | relative_url }}" alt="Color Identity" title="Color Identity Distribution" width="500"/>
</div>

## Power & Toughness
I looked at all the numeric power and toughness values of cards. I excluded "X/X" cards for now. You can see that there are a lot more cards with a power of 0 than cards with a toughness of 0 (Disclaimer: there are 0/0 cards in MTG).  
The higher number of 0/X cards than X/0 cards are mainly "Defender" cards that can't attack anyways.

<div style="text-align: center">
    <img src="{{ '/assets/img/power_toughness_distribution.png' | relative_url }}" alt="Power Toughness Distribution" title="Power Toughness Distribution" width="500"/>
</div>

## Rarity
Rarity is an important property of a card in MTG. We have four rarities: Common (black), Uncommon (silver), Rare (gold) and Mythic (orange). 

<div style="text-align: center">
    <img src="{{ '/assets/img/rarity_distribution.png' | relative_url }}" alt="Rarity Distribution" title="Rarity Distribution" width="500"/>
</div>


## Keywords
As in most TCGs, cards in MTG have certain powers which are described and mentioned in the so-called oracle text of a card.  
Certain powers reappear on multiple cards and have keywords to not always include the whole description of the power.  
"Haste" for example allows a creature to attack in the same turn as it was played. A creature without haste has to "wait" one turn to attack.  
The following table shows the percentages of some of the most important keywords of MTG cards.

| Keyword | Percentage |
| -------- | ------- |
| Lifelink | 2.08% |
| Trample | 5.12% |
| Flying | 19.18% |
| Haste | 3.75% |
| First Strike | 2.27% |
| Double Strike | 0.61% |
| Deathtouch | 1.90% |
| Vigilance | 3.60% |

## Prices
As most TCGs, magic cards are sold on various websites and markets and have value corresponding to their demand on the market. Given the lower probability of getting a rare or mythic card from a booster pack we can assume that these cards are also more valuable.  
I looked at the prices with respect to the rarities in the dataset which confirms this hypothesis.  
Median and mean clearly show how rares and mythics are more expensive than common and uncommon. Interestingly, we can see that there seem to be some very expensive rares and mythics which lead to a pretty big standard deviation for these rarities.  
The maximum card prices was around 6000$ in this dataset.

| Metric | Rarity    | Price |
| -------- | -------  | ------- |
| Median | Common | 0.11$
|  | Uncommon | 0.18$
| | Rare| 0.66$
| | Mythic | 3.04$
| Mean | Common | 0.79$
| | Uncommon | 2.20$
| | Rare| 7.61$
| | Mythic | 8.55$
| 95th percentile | Common | 13.35$
| | Uncommon | 33.40$
| | Rare | 103.09$
| | Mythic | 72.39$

<div style="text-align: center">
    <img src="{{ '/assets/img/price_distribution.png' | relative_url }}" alt="Price Distribution" title="Price Distribution" width="800"/>
</div>

<div style="text-align: center">
    <img src="{{ '/assets/img/price_distribution_by_rarity.png' | relative_url }}" alt="Price Distribution by Rarity" title="Price Distribution by Rarity" width="500"/>
</div>