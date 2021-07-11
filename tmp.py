import gtts

tts = gtts.tts.gTTS(text='Hello', lang='en')
tts.save("hello.mp3")