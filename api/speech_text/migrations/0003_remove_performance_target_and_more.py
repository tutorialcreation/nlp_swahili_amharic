# Generated by Django 4.0.4 on 2022-06-07 04:35

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('speech_text', '0002_performance_wer_rate'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='performance',
            name='target',
        ),
        migrations.RemoveField(
            model_name='performance',
            name='wer_rate',
        ),
        migrations.AddField(
            model_name='performance',
            name='audio',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='speech_text.audio'),
        ),
    ]
