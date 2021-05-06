#%% Use selenium (a fake browser to scrape all data)
# import and set up general stuff
from selenium import webdriver
import time
import json
import tqdm
driver = webdriver.Chrome()



#%% Get the main page and all sub links
URL = 'https://chordify.net/chords/artist/taylor-swift'
driver.get(URL)

div = driver.find_element_by_xpath('/html/body/div[2]/main/div/div[3]/section/div[2]')
a_elements = div.find_elements_by_tag_name('a')
pages_to_search = []
for page in a_elements:
    # chordify link
    page_obj = {'link': page.get_attribute("href")}
    # chordify song title
    span_name =page.find_elements_by_tag_name('span')[0]
    title = str(span_name.text)
    page_obj['title'] = title
    pages_to_search.append(page_obj)
print('All '+str(len(pages_to_search))+' songs found')

#%% Navigate each page ... takes about 10 mins
# tqdm wrapper just shows a bar and estimated time to finish
# tqdm
for page_index, page in enumerate(pages_to_search):
    driver.get( page['link'] )
    time.sleep(1.5)
    # get youtube link to get number of views after
    # youtube_div = driver.find_element_by_class_name('view-diagrams')
    # youtube_link = youtube_div.get_attribute('data-stream')
    # pages_to_search[page_index]['youtube_link'] = youtube_link

    # Get key of song
    toolbar_single_item_ul = driver.find_element_by_id('edit-toolbar-legacy')
    key_span = toolbar_single_item_ul.find_element_by_xpath('li[1]/span[2]/a/span/span')
    key = key_span.get_attribute('class')
    pages_to_search[page_index]['key'] = key
    print(page_index)

    # get chords array
    chords = []
    # chords_div = driver.find_element_by_id('chords')
    # chords_array = chords_div.find_elements_by_tag_name('div')
    # for chord in chords_array:
    #     classes = chord.get_attribute('class').split()
    #     if ('nolabel' not in classes) and ('label-wrapper' not in classes) and type(chord.get_attribute('data-i'))==str:
    #         obj ={}
    #         # get index
    #         # print(type(chord.get_attribute('data-i')))
    #         data_i = int(chord.get_attribute('data-i'))
    #         obj['data_i'] = data_i
    #         # get chord
    #         chord_label_element = chord.find_element_by_class_name('chord-label')
    #         chord_label = chord_label_element.get_attribute('class').split()[1]
    #         obj['chord'] = chord_label
    #         # append to chords
    #         chords.append(obj)
    # pages_to_search[page_index]['chords'] = chords
print('All chords found')

# %% Dump all that knowledge into a json file
with open("data/names.json", 'w') as outfile:
    json.dump(pages_to_search, outfile)
print('Data exported to data/keys.json')

