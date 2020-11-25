from aip import AipSpeech
import pyaudio
import wave



def get_audio(filePath):
    aa=str(input("是否开始录音?     (y/n)"))

    if aa==str("y"):
        CHUNK=1024
        FORMAT=pyaudio.paInt16
        CHANNELS=1
        RATE=11025
        RECORD_SECONDS=10
        WAVE_OUTPUT_FILENAME=filePath
        p=pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print("*"*5, "开始录音：请在10秒内输入语音", "*"*5)
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("*"*5, "录音结束\n")  
        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    elif aa == str("n"):
        exit()
    else:
        print("语音录入失败，请重新开始")
        get_audio(in_path)

def get_file_content(filePath):
    with open(filePath,'rb') as fp:
        return fp.read()


if __name__=="__main__":
    
    APP_ID= "23038327"

    API_KEY= "7RwjGAYYTHXBKWo4dcP3Aaga"

    SECRET_KEY="bonCBakFEuCTL4SjQkbtdBIF38XG3dEB"

    Clinet= AipSpeech(APP_ID,API_KEY,SECRET_KEY)
    in_path="./audios/input.wav"
    get_audio(in_path)
    '''
    1537 普通话
    1737 英语
    '''
    result=Clinet.asr(get_file_content(in_path),'wav',16000,{'dev_pid':1537,})
    #print(result)
    print(result['result'][0])

