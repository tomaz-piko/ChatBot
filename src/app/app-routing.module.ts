import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { ChatBoxComponent } from './chat-box/chat-box.component';
import { ChatBoxEngComponent } from './chat-box-eng/chat-box-eng.component';

const routes: Routes = [
  {path: '', component: ChatBoxComponent },
  {path: 'ChatBoxEng', component: ChatBoxEngComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
