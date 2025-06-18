import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

def send_email(content):
    # SMTP 서버와 포트
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    subject = f'{datetime.now().strftime("%Y-%m-%d")} 위험 가구 이상지 탐지'
    body = content
    to_email = 'jeong235711@gmail.com'
    from_email = "program.mark1"
    password = "teaz wcxz acwt isry"
    # MIME 메시지 생성
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # 이메일 본문 추가
    msg.attach(MIMEText(body, 'plain'))

    # SMTP 서버에 연결하고 이메일 전송
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # TLS 암호화 시작
            server.login(from_email, password)  # 로그인
            server.send_message(msg)  # 이메일 전송
            print('이메일이 성공적으로 전송되었습니다.')
            return True

    except Exception as e:
        print(f'이메일 전송 중 오류 발생: {e}')
        return False

