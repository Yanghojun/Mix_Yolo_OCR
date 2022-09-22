import playsound

korean_list = {'보가드': './mp3_dir/보가드.m4a', 
            '근초대왕':'./mp3_dir/근초대왕.m4a', 
               '꽉자바': './mp3_dir/깍자바.m4a', 
               '팬텀': './mp3_dir/팬텀.m4a',
               '팬럼': './mp3_dir/팬텀.m4a',
               '슈퍼펀지':'./mp3_dir/슈퍼펀치.m4a',
               '슈퍼편지':'./mp3_dir/슈퍼펀치.m4a',
               'cleanser': './mp3_dir/Cleanser.m4a',
               }

def play_my_sweet_voice(ele:list)->None:
    print(ele)
    for key, mp3_path in korean_list.items():
        if key in ele:
            playsound.playsound(mp3_path)
            break
