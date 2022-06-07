from django.db import models

# Create your models here.
class AudioParams(models.Model):
    key = models.CharField(max_length=255,null=True,blank=True)
    text = models.TextField(null=True,blank=True)
    duration = models.FloatField(null=True,blank=True)
    rate = models.FloatField(null=True,blank=True)
    file=models.URLField(null=True,blank=True)
    rmse=models.FloatField(null=True,blank=True)
    chroma_stft=models.FloatField(null=True,blank=True)
    spec_cent=models.FloatField(null=True,blank=True)
    spec_bw=models.FloatField(null=True,blank=True)
    rolloff=models.FloatField(null=True,blank=True)
    zcr=models.FloatField(null=True,blank=True)
    

    def __str__(self) -> str:
        return self.key

    @property
    def get_mfcc(self):
        pass

class Audio(models.Model):
    audio_file = models.FileField(upload_to="media/audio/",null=True,blank=True)

class Performance(models.Model):
    audio = models.ForeignKey(Audio,on_delete=models.CASCADE,null=True,blank=True)
    prediction=models.TextField(null=True,blank=True)


    