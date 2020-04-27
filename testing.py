'''
Script for testing different aspects of the model
'''

from interface import predict_youtube

'''
Generalization testing on YouTube videos
'''

reals = [
    'https://www.youtube.com/watch?v=mfT-UZCA6Tg',
    'https://www.youtube.com/watch?v=jNQXAC9IVRw',
    'https://www.youtube.com/watch?v=a9J8GaeDqVc',
    'https://www.youtube.com/watch?v=y7hddyiR47k',
    'https://www.youtube.com/watch?v=SzBFR2EE8hM',
    'https://www.youtube.com/watch?v=a2fhXo4SzfA',
    'https://www.youtube.com/watch?v=oBM7DIeMsP0',
    'https://www.youtube.com/watch?v=36zrJfAFcuc',

]
fakes = [
    'https://www.youtube.com/watch?v=AQvCmQFScMA',
    'https://www.youtube.com/watch?v=Gz0QZP2RKWA',
    'https://www.youtube.com/watch?v=l_6Tumd8EQI',
    'https://www.youtube.com/watch?v=Ho9h0ouemWQ',
    'https://www.youtube.com/watch?v=cVljNVV5VPw',
    'https://www.youtube.com/watch?v=YjwOxPKJpaQ',
    'https://www.youtube.com/watch?v=XXtl5wnSR4M',
]

real_count, fake_count = 0,0
real_correct, fake_correct = 0,0
for label, urls in enumerate([fakes, reals]):
    print('REALS' if label == 1 else 'FAKES')
    for url in urls:
        prediction, confidence = predict_youtube(url)
        print(f'{url}: {prediction} {round(confidence,2)}%')
        if prediction in [1,0]:
            if label == 1:
                real_count += 1
                real_correct += int(label == prediction)
            if label == 0:
                fake_count += 1
                fake_correct += int(label == prediction)

print(f'True positives={real_correct/real_count}')
print(f'True negatives={fake_correct/fake_count}')
print(f'Accuracy={(real_correct+fake_correct)/(real_count+fake_count)}')