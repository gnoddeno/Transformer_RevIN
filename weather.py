import pandas as pd
import urllib.request
import urllib.parse
import json

from datetime import datetime, timedelta

def get_tomorrow_date(date: str = None) -> str:
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    date_obj = datetime.strptime(date, '%Y%m%d')
    next_date = date_obj + timedelta(days=1)
    return next_date.strftime('%Y%m%d')

# ▶️ 1. 날짜 설정
def weather_api(date :str = None):
    target_date = get_tomorrow_date(date)  
    base_date = date    
    base_time = '0500'        

    # ▶️ 2. API 요청 URL 구성
    servicekey = 'qm+twia1GdNQoIZKIJKpIbJK5JQlFwLcmur5ObEM+SyC6o27zWSWdn0XH2nwGCY1FGm3A5cT7paDgIVzXj9XCA=='
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
    queryParams = '?' + urllib.parse.urlencode({
        'ServiceKey': servicekey,
        'numOfRows': '520',
        'dataType': 'JSON',
        'base_date': base_date,
        'base_time': base_time,
        'nx': '102',
        'ny': '83'
    })

    response = urllib.request.urlopen(url + queryParams).read()
    response = json.loads(response)
    items = response['response']['body']['items']['item']


    target_hours = [f"{h:02}00" for h in range(24)]
    columns = ['Forecast_time', 'temperature', 'humidity', 'precipitation', 'windspeed']
    fcst_df = pd.DataFrame(columns=columns)
    fcst_df['Forecast_time'] = [f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]} {h:02}:00" for h in range(24)]


    data_by_time = {hour: {} for hour in target_hours}

    for data in items:
        if data['fcstDate'] != target_date:
            continue  # ✅ 내가 알고자 하는 날짜의 데이터만 수집

        fcst_time = data['fcstTime']
        category = data['category']
        value = data['fcstValue']

        if fcst_time not in data_by_time:
            continue

        if category == 'TMP':
            data_by_time[fcst_time]['temperature'] = float(value)
        elif category == 'REH':
            data_by_time[fcst_time]['humidity'] = float(value)
        elif category == 'WSD':
            data_by_time[fcst_time]['windspeed'] = float(value)
        elif category == 'PCP':
            if value == '강수없음':
                data_by_time[fcst_time]['precipitation'] = 0.0
            elif 'mm' in value:
                data_by_time[fcst_time]['precipitation'] = float(value.replace('mm', ''))
            else:
                try:
                    data_by_time[fcst_time]['precipitation'] = float(value)
                except ValueError:
                    data_by_time[fcst_time]['precipitation'] = 0.0

    # ▶️ 6. DataFrame에 삽입
    for i, hour in enumerate(target_hours):
        for col in columns[1:]:
            fcst_df.loc[i, col] = data_by_time.get(hour, {}).get(col)

    # ▶️ 7. 결측값 0으로 채우기
    fcst_df = fcst_df.fillna(0)

    # ▶️ 8. 결과 출력
    print(fcst_df)
    return fcst_df
