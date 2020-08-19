import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { ChatBoxEngComponent } from './chat-box-eng.component';

describe('ChatBoxEngComponent', () => {
  let component: ChatBoxEngComponent;
  let fixture: ComponentFixture<ChatBoxEngComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ChatBoxEngComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ChatBoxEngComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
